"""Public planning API -- what later orchestration code calls.

    spec = inspect_model("gpt2")
    workers = [from_local_machine("computer-1"), from_simulated_profile(path)]
    plan = create_plan(spec, workers, precision=Precision("fp32"))

No REST, gRPC, worker-process or frontend coupling: inputs are plain
data (ModelSpec, WorkerResources), output is a JSON-ready dict (schema:
docs/PARTITION_PLAN_SCHEMA.md). Every returned plan has already passed
validate_plan(); an internal inconsistency raises instead of returning
a bad plan.
"""

from __future__ import annotations

import datetime as _dt
import itertools
import socket
from collections import Counter
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from partitioner import model_spec as ms
from partitioner import plan as plan_mod
from partitioner.cost import CostModel, ExecutionTarget, WorkerOverride, resolve_target, transfer_seconds, unit_flops
from partitioner.memory import MemoryEstimator, MemoryPolicy, Precision, StageMemory, Workload
from partitioner.optimizer import solve
from partitioner.plan import fmt_bytes
from partitioner.workers import REAL, SIMULATED, WorkerResources


class PlanningError(ValueError):
    pass


def _fits(target: ExecutionTarget, mem: StageMemory) -> bool:
    if target.device == "cpu":
        return mem.total_bytes <= target.device_capacity_bytes
    return mem.device_bytes <= target.device_capacity_bytes and mem.host_bytes <= target.host_capacity_bytes


def _misfit(target: ExecutionTarget, mem: StageMemory) -> str:
    if target.device == "cpu":
        return f"needs {fmt_bytes(mem.total_bytes)} RAM > {fmt_bytes(target.device_capacity_bytes)} effective"
    parts = []
    if mem.device_bytes > target.device_capacity_bytes:
        parts.append(f"needs {fmt_bytes(mem.device_bytes)} VRAM > {fmt_bytes(target.device_capacity_bytes)}")
    if mem.host_bytes > target.host_capacity_bytes:
        parts.append(f"needs {fmt_bytes(mem.host_bytes)} host RAM > {fmt_bytes(target.host_capacity_bytes)}")
    return "; ".join(parts)


class _Problem:
    """Precomputed stage memory (worker-independent) and costs."""

    def __init__(self, spec, estimator: MemoryEstimator, targets: List[ExecutionTarget], cost_model: CostModel, precision):
        self.spec, self.est, self.targets = spec, estimator, targets
        self.U = len(spec.units)
        flops = [unit_flops(spec, u, estimator.workload) for u in spec.units]
        self.prefix = [0.0]
        for f in flops:
            self.prefix.append(self.prefix[-1] + f)
        self.transfer = transfer_seconds(spec, estimator.workload, precision, cost_model)
        self._mem: Dict[Tuple[int, int], StageMemory] = {}

    def mem(self, i: int, j: int) -> StageMemory:
        key = (i, j)
        if key not in self._mem:
            self._mem[key] = self.est.stage(i, j)
        return self._mem[key]

    def compute_seconds(self, t: ExecutionTarget, i: int, j: int) -> float:
        return (self.prefix[j] - self.prefix[i]) / t.flops_per_s

    def cost(self, t: ExecutionTarget, i: int, j: int) -> Optional[float]:
        if not _fits(t, self.mem(i, j)):
            return None
        seconds = self.compute_seconds(t, i, j)
        if self.transfer is not None and j < self.U:
            seconds += self.transfer
        return seconds


def create_plan(
    spec: ms.ModelSpec,
    workers: Sequence[WorkerResources],
    precision: Precision = Precision(),
    workload: Workload = Workload(),
    memory_policy: MemoryPolicy = MemoryPolicy(),
    cost_model: CostModel = CostModel(),
    overrides: Optional[Mapping[str, WorkerOverride]] = None,
    allow_simulated: bool = True,
    allow_unknown_source: bool = False,
    allow_unused_workers: bool = True,
    permutation_limit: int = 6,
    now: Optional[_dt.datetime] = None,
) -> dict:
    cost_model.validate()
    estimator = MemoryEstimator(spec, precision, workload, memory_policy)
    wl = estimator.workload
    overrides = dict(overrides or {})
    ids = [w.worker_id for w in workers]
    dupes = sorted(i for i, n in Counter(ids).items() if n > 1)
    if dupes:
        raise PlanningError(f"duplicate worker ids: {dupes}")
    unknown_overrides = sorted(set(overrides) - set(ids))
    if unknown_overrides:
        raise PlanningError(f"overrides given for unknown workers: {unknown_overrides}")

    warnings: List[str] = []
    assumptions: List[str] = [
        "Stage time = calculated FLOPs / effective throughput; elementwise ops and memory-bandwidth limits are not modelled.",
        "Memory covers a worst-case prefill of prompt_length tokens with a KV cache sized for context_length.",
        "Activations assume eager attention (scores materialized) and are freed layer by layer.",
        "The first stage holds the embeddings, the last stage the final norm and LM head (layer order is fixed).",
    ]
    if cost_model.link_bandwidth_bytes_per_s is None:
        warnings.append("Inter-stage transfer time is NOT modelled (no link bandwidth given); real pipelines pay it per boundary.")
    if precision.quantization != "none":
        warnings.append(
            f"{precision.quantization} memory is an ESTIMATE of what a quantized runtime would need; "
            "no quantized execution is implemented in this project."
        )
    if precision.weight_dtype in ("fp16", "bf16") or precision.act in ("fp16", "bf16"):
        warnings.append(
            f"{precision.weight_dtype}/{precision.act} on CPU workers: supported by PyTorch but typically slower than fp32 "
            "on CPUs without native support; the cost model does not account for this."
        )

    # ---- eligibility ------------------------------------------------
    unused: List[dict] = []
    candidates: List[WorkerResources] = []
    for w in sorted(workers, key=lambda w: w.worker_id):
        if w.profile_source == SIMULATED and not allow_simulated:
            unused.append(_unused(w, "SIMULATED profile and allow_simulated is false"))
        elif w.profile_source not in (REAL, SIMULATED) and not allow_unknown_source:
            unused.append(_unused(w, f"profile source {w.profile_source!r} is neither REAL nor SIMULATED"))
        else:
            candidates.append(w)

    # Several REAL workers on one machine share its RAM: split it.
    host_counts = Counter(w.hostname for w in candidates if w.profile_source == REAL and w.hostname and w.online)
    targets: List[ExecutionTarget] = []
    for w in candidates:
        share = 1.0
        if w.profile_source == REAL and w.hostname and host_counts.get(w.hostname, 0) > 1:
            share = 1.0 / host_counts[w.hostname]
            warnings.append(
                f"{w.worker_id} shares host {w.hostname!r} with {host_counts[w.hostname] - 1} other worker(s); "
                f"each is given 1/{host_counts[w.hostname]} of its usable RAM. Local processes on one machine are "
                "NOT independent computers."
            )
        target, tw, reason = resolve_target(w, overrides.get(w.worker_id, WorkerOverride()), cost_model, memory_policy, share)
        warnings.extend(tw)
        if target is None:
            unused.append(_unused(w, reason))
        else:
            targets.append(target)
            if "micro-benchmark" in target.throughput_basis:
                warnings.append(f"{w.worker_id} throughput comes from a single noisy micro-benchmark: {target.throughput_basis}.")
            else:
                assumptions.append(f"{w.worker_id} throughput: {target.throughput_basis}.")

    this_host = socket.gethostname()
    for w in candidates:
        if w.profile_source == REAL and w.hostname == this_host:
            warnings.append(
                f"{w.worker_id} is this planning machine ({this_host}): its available RAM was measured while this "
                "planner process (~700 MiB, mostly torch + transformers) was running, so it is lower than a worker "
                "would see without the planner."
            )

    includes_simulated = any(t.worker.profile_source == SIMULATED for t in targets)
    if includes_simulated:
        warnings.append(
            "This plan includes SIMULATED - NOT PHYSICALLY VERIFIED workers; it describes hypothetical hardware "
            "and must not be presented as a plan for real machines."
        )

    base = _plan_base(spec, precision, wl, memory_policy, cost_model, estimator, workers, now)
    problem = _Problem(spec, estimator, targets, cost_model, precision) if targets else None

    best = None
    tried = 0
    if problem is not None:
        if len(targets) <= permutation_limit:
            orders = itertools.permutations(targets)
            order_note = f"all {_factorial(len(targets))} worker orders searched"
        else:
            orders = [tuple(targets)]
            order_note = f"{len(targets)} workers > permutation_limit {permutation_limit}: worker-id order used, not searched"
            warnings.append(order_note)
        for order in orders:
            tried += 1
            sol = solve(lambda k, i, j, o=order: problem.cost(o[k], i, j), problem.U, len(order), allow_unused_workers)
            if sol is None:
                continue
            if best is None or _better(sol, best[0]):
                best = (sol, order)
    else:
        order_note = "no usable workers"

    if best is None:
        plan = dict(base)
        plan.update(
            feasible=False,
            objective=_objective(None, wl, order_note),
            stages=[],
            unused_workers=unused + [_unused(t.worker, "not used: no feasible plan") for t in targets],
            includes_simulated_workers=includes_simulated,
            executable=False,
            executable_reason="infeasible",
            warnings=warnings,
            assumptions=assumptions,
            infeasibility={"reasons": _diagnose(spec, estimator, targets, unused, problem)},
        )
        plan["inputs_fingerprint"] = plan_mod.fingerprint(plan)
        return plan

    sol, order = best
    stages = []
    used_ids = set()
    for k, ((s, e), cost) in enumerate(zip(sol.ranges, sol.stage_costs)):
        if s == e:
            continue
        t = order[k]
        used_ids.add(t.worker.worker_id)
        stages.append(_stage(spec, problem, t, s, e, cost, sol.bottleneck, len(stages)))
    for t in targets:
        if t.worker.worker_id in used_ids:
            continue
        fits_any = any(_fits(t, problem.mem(i, i + 1)) for i in range(problem.U))
        if not fits_any:
            smallest = min(problem.mem(i, i + 1).total_bytes for i in range(problem.U))
            reason = (
                f"too little memory for any stage: effective capacity {fmt_bytes(t.device_capacity_bytes)} "
                f"< smallest single-unit stage {fmt_bytes(smallest)} (runtime overhead included)"
            )
        else:
            reason = (
                f"eligible ({t.device}, {t.flops_per_s / 1e9:.1f} effective GFLOPS: {t.throughput_basis}) but not "
                "needed: the optimal bottleneck is reached without it"
            )
        unused.append(_unused(t.worker, reason))
    for stage, nxt in zip(stages, stages[1:]):
        stage["explanation"].extend(_boundary_sensitivity(problem, stages, stage, nxt, order, sol.bottleneck))

    from partitioner.execution.models import executable_status

    executable, executable_reason = executable_status(spec.model_id, precision, [s["profile_source"] for s in stages])
    plan = dict(base)
    plan.update(
        feasible=True,
        objective=_objective(sol, wl, f"{order_note} ({tried} evaluated); ties -> lower total time, then worker-id order"),
        stages=stages,
        unused_workers=sorted(unused, key=lambda u: u["worker_id"]),
        includes_simulated_workers=any(s["profile_source"] == SIMULATED for s in stages),
        executable=executable,
        executable_reason=executable_reason,
        warnings=warnings,
        assumptions=assumptions,
        infeasibility=None,
    )
    plan["inputs_fingerprint"] = plan_mod.fingerprint(plan)

    from partitioner.validate import validate_plan

    validate_plan(plan, spec, {w.worker_id: w for w in workers}, estimator=estimator)
    return plan


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def _factorial(n: int) -> int:
    out = 1
    for i in range(2, n + 1):
        out *= i
    return out


def _better(a, b) -> bool:
    tol = 1e-12 * max(1.0, abs(b.bottleneck))
    if a.bottleneck < b.bottleneck - tol:
        return True
    if a.bottleneck > b.bottleneck + tol:
        return False
    return a.total < b.total - 1e-12 * max(1.0, abs(b.total))


def _unused(w: WorkerResources, reason: str) -> dict:
    return {
        "worker_id": w.worker_id,
        "profile_source": w.profile_source,
        "reason": reason,
        # The figures the decision was based on, so "unused" is checkable.
        "ram_total_bytes": w.ram_total_bytes,
        "ram_available_bytes": w.ram_available_bytes,
        "ram_usable_bytes": w.ram_usable_bytes,
    }


def _plan_base(spec, precision, wl, policy, cost_model, estimator, workers, now) -> dict:
    created = (now or _dt.datetime.now(_dt.timezone.utc)).isoformat()
    return {
        "partition_plan_schema_version": plan_mod.PLAN_SCHEMA_VERSION,
        "created_at": created,
        "planner": {"name": plan_mod.PLANNER_NAME, "version": plan_mod.PLANNER_VERSION, "algorithm": plan_mod.ALGORITHM},
        "model": {
            **spec.summary(),
            "config": spec.config,
            "checkpoint": spec.checkpoint.to_dict() if spec.checkpoint else None,
            "units": [
                {"index": u.index, "name": u.name, "kind": u.kind, "components": list(u.components), "block_index": u.block_index}
                for u in spec.units
            ],
            "unit_memory": estimator.unit_table(),
        },
        "precision": precision.to_dict(),
        "workload": {
            "batch_size": wl.batch_size,
            "context_length": wl.context_length,
            "prompt_length": wl.prompt_length,
            "cost_mode": wl.cost_mode,
            "logits_positions": wl.logits_positions,
        },
        "memory_policy": policy.to_dict(),
        "cost_model": {
            "cpu_flops_per_cycle_per_core": cost_model.cpu_flops_per_cycle_per_core,
            "throughput_preference": cost_model.throughput_preference,
            "account_for_utilization": cost_model.account_for_utilization,
            "link_bandwidth_bytes_per_s": cost_model.link_bandwidth_bytes_per_s,
            "link_latency_s": cost_model.link_latency_s,
        },
        "workers_considered": [
            {"worker_id": w.worker_id, "profile_source": w.profile_source, "hostname": w.hostname,
             "simulated_profile_name": w.simulated_profile_name, "origin": w.origin}
            for w in sorted(workers, key=lambda w: w.worker_id)
        ],
    }


def _objective(sol, wl, note: str) -> dict:
    return {
        "type": "minimize_max_stage_time",
        "cost_basis": f"{wl.cost_mode} step (batch {wl.batch_size}, context {wl.context_length})",
        "bottleneck_seconds": sol.bottleneck if sol else None,
        "total_seconds": sol.total if sol else None,
        "seconds_provenance": "ESTIMATED",
        "worker_order_search": note,
    }


def _stage(spec, problem: _Problem, t: ExecutionTarget, s: int, e: int, cost: float, bottleneck: float, index: int) -> dict:
    units = spec.units[s:e]
    mem = problem.mem(s, e)
    blocks = [u.block_index for u in units if u.block_index is not None]
    components = [c for u in units for c in u.components]
    pct_cap = 100.0 * (mem.total_bytes if t.device == "cpu" else mem.device_bytes) / max(1, t.device_capacity_bytes)
    is_bottleneck = abs(cost - bottleneck) <= 1e-12 * max(1.0, bottleneck)
    compute = problem.compute_seconds(t, s, e)
    transfer = cost - compute
    time_note = (
        f"Estimated {cost * 1e3:.3f} ms per step: {compute * 1e3:.3f} ms compute at {t.flops_per_s / 1e9:.1f} effective "
        f"GFLOPS ({t.throughput_basis}) + {transfer * 1e3:.3f} ms sending its output"
        if transfer > 0
        else f"Estimated {cost * 1e3:.3f} ms per step at {t.flops_per_s / 1e9:.1f} effective GFLOPS ({t.throughput_basis})"
    )
    runs: List[list] = []  # in model order: [name] or [first_block, last_block]
    for u in units:
        if u.block_index is None:
            runs.append([u.name])
        elif runs and len(runs[-1]) == 2 and runs[-1][1] == u.block_index - 1:
            runs[-1][1] = u.block_index
        else:
            runs.append([u.block_index, u.block_index])
    names = ", ".join(
        r[0] if len(r) == 1 else (f"block {r[0]}" if r[0] == r[1] else f"blocks {r[0]}-{r[1]}") for r in runs
    )
    expl = [
        f"Runs units {s}..{e - 1} ({names}) on {t.device}/{t.backend}.",
        (
            f"Memory: needs {fmt_bytes(mem.total_bytes)} ({mem.provenance.value}) of {fmt_bytes(t.device_capacity_bytes)} "
            f"effective RAM ({pct_cap:.1f}%)."
            if t.device == "cpu"
            else f"Memory: {fmt_bytes(mem.device_bytes)} VRAM of {fmt_bytes(t.device_capacity_bytes)} ({pct_cap:.1f}%), "
                 f"{fmt_bytes(mem.host_bytes)} host RAM of {fmt_bytes(t.host_capacity_bytes)}."
        ),
        time_note
        + (" -- this is the BOTTLENECK stage." if is_bottleneck else f" -- {100 * cost / bottleneck:.1f}% of the bottleneck."),
    ]
    if any(i.name == "tied_weight_copy" for i in mem.items):
        expl.append("Holds its own copy of the LM head weight: it is tied to the token embedding, which lives on another stage.")
    w = t.worker
    return {
        "stage_index": index,
        "worker_id": w.worker_id,
        "profile_source": w.profile_source,
        "simulated_profile_name": w.simulated_profile_name or None,
        "hostname": w.hostname,
        "unit_range": [s, e],
        "layer_range": {"first_block": blocks[0], "last_block": blocks[-1]} if blocks else None,
        "units": [u.name for u in units],
        "components": components,
        "execution": t.to_dict(),
        "memory": mem.to_dict(),
        "memory_fits": _fits(t, mem),
        "available_memory": {
            "ram_available_bytes": w.ram_available_bytes,
            "ram_usable_bytes": w.ram_usable_bytes,
            "ram_phase3_margin_bytes": w.ram_safety_margin_bytes,
            "device_capacity_bytes": t.device_capacity_bytes,
            "host_capacity_bytes": t.host_capacity_bytes,
        },
        "estimated_seconds": cost,
        "estimated_compute_seconds": compute,
        "estimated_transfer_seconds": transfer,
        "estimated_flops": problem.prefix[e] - problem.prefix[s],
        "cost_provenance": "ESTIMATED",
        "is_bottleneck": is_bottleneck,
        "explanation": expl,
    }


def _boundary_sensitivity(problem: _Problem, stages, a: dict, b: dict, order, bottleneck: float) -> List[str]:
    """What happens if the boundary between stage a and b moves by one
    unit in either direction -- shows the split is locally optimal."""
    by_id = {t.worker.worker_id: t for t in order}
    ta, tb = by_id[a["worker_id"]], by_id[b["worker_id"]]
    (sa, ea), (sb, eb) = a["unit_range"], b["unit_range"]
    others = max([s["estimated_seconds"] for s in stages if s is not a and s is not b], default=0.0)
    out = []
    for shift, label in ((-1, "one unit earlier"), (1, "one unit later")):
        nb = ea + shift
        if not (sa < nb < eb):
            continue
        ca, cb = problem.cost(ta, sa, nb), problem.cost(tb, nb, eb)
        moved = problem.spec.units[min(ea, nb)].name
        if ca is None or cb is None:
            who, (t, s, e) = (a["worker_id"], (ta, sa, nb)) if ca is None else (b["worker_id"], (tb, nb, eb))
            out.append(f"Moving the {a['worker_id']}->{b['worker_id']} boundary {label} ({moved}) is infeasible: "
                       f"{who} {_misfit(t, problem.mem(s, e))}.")
        else:
            nbott = max(others, ca, cb)
            out.append(f"Moving the {a['worker_id']}->{b['worker_id']} boundary {label} ({moved}) would give a "
                       f"bottleneck of {nbott * 1e3:.3f} ms ({100 * (nbott - bottleneck) / bottleneck:+.2f}%).")
    return out


def _diagnose(spec, est: MemoryEstimator, targets, unused, problem) -> List[str]:
    reasons = []
    if not targets:
        reasons.append("No usable workers.")
        reasons.extend(f"{u['worker_id']}: {u['reason']}" for u in unused)
        return reasons
    U = len(spec.units)
    whole = est.stage(0, U)
    caps = sorted(((t.device_capacity_bytes, t.worker.worker_id) for t in targets), reverse=True)
    reasons.append(
        f"The whole model on one worker needs {fmt_bytes(whole.total_bytes)}; the largest effective capacity is "
        f"{fmt_bytes(caps[0][0])} ({caps[0][1]}); all usable workers together have {fmt_bytes(sum(c for c, _ in caps))}."
    )
    single = [(est.stage(i, i + 1).total_bytes, spec.units[i].name) for i in range(U)]
    worst_bytes, worst_unit = max(single)
    if worst_bytes > caps[0][0]:
        reasons.append(
            f"Unit {worst_unit} alone needs {fmt_bytes(worst_bytes)} (including the per-worker runtime overhead), "
            f"more than any worker can hold."
        )
    reasons.append(
        f"Every stage also pays a {fmt_bytes(est.policy.runtime_overhead_bytes)} runtime overhead, so adding workers "
        "adds memory as well as capacity."
    )
    reasons.extend(f"Unusable: {u['worker_id']}: {u['reason']}" for u in unused)
    reasons.append(
        "Options: shorter context_length/prompt_length, logits_positions='last', fp16/bf16 weights, or more/larger workers."
    )
    return reasons
