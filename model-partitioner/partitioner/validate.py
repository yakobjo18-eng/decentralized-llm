"""Partition plan validator.

Independent of the optimizer: it checks a plan dict (e.g. loaded from
JSON, possibly hand-edited) against the model spec and, if given, the
worker resources and a memory estimator. Collects EVERY problem and
raises PlanValidationError listing them all.

Checks:
  schema version; model fingerprint/architecture/unit count match;
  stages ordered 0..k-1; unit ranges non-empty, contiguous, starting at
  0, ending at the last unit (=> every layer assigned, none twice,
  order preserved); listed units/components/layer_range agree with the
  model; every model component assigned exactly once; worker ids
  unique, known, and not also listed as unused; profile sources match
  the worker records; includes_simulated_workers is truthful; memory
  fits each stage's stated capacity; and, with an estimator, the
  stated memory requirement matches a recomputation (so a plan with an
  understated requirement is rejected).
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Mapping, Optional

from partitioner import model_spec as ms
from partitioner.optimizer import check_ranges
from partitioner.plan import SUPPORTED_SCHEMA_VERSIONS


class PlanValidationError(ValueError):
    def __init__(self, problems: List[str]):
        self.problems = problems
        super().__init__("invalid partition plan:\n  - " + "\n  - ".join(problems))


def estimator_for_plan(plan: dict, spec: ms.ModelSpec):
    """Rebuilds the MemoryEstimator from the settings recorded in the
    plan, so validate_plan can recompute every stage's requirement.
    Raises KeyError/TypeError/EstimationError if the plan lacks them."""
    from partitioner.memory import MemoryEstimator, MemoryPolicy, Precision, Workload

    pr = plan["precision"]
    precision = Precision(**{k: pr[k] for k in ("weight_dtype", "quantization", "quant_group_size",
                                                "quantize_embeddings", "quantize_lm_head")},
                          activation_dtype=pr["activation_dtype"], kv_cache_dtype=pr["kv_cache_dtype"])
    mp = plan["memory_policy"]
    policy = MemoryPolicy(runtime_overhead_bytes=mp["runtime_overhead_bytes"], extra_safety_fraction=mp["extra_safety_fraction"],
                          include_kv_cache=mp["include_kv_cache"], include_load_temp=mp["include_load_temp"],
                          checkpoint_dtype=mp["checkpoint_dtype"])
    return MemoryEstimator(spec, precision, Workload(**plan["workload"]), policy)


def validate_plan(
    plan: dict,
    spec: ms.ModelSpec,
    workers: Optional[Mapping[str, object]] = None,
    estimator=None,
) -> None:
    p: List[str] = []
    version = plan.get("partition_plan_schema_version")
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise PlanValidationError([f"unsupported partition_plan_schema_version {version!r} (supported: {SUPPORTED_SCHEMA_VERSIONS})"])

    model = plan.get("model") or {}
    if model.get("config_fingerprint") != spec.config_fingerprint:
        p.append("plan was made for a different model configuration (config_fingerprint mismatch)")
    if model.get("architecture") != spec.architecture:
        p.append(f"plan architecture {model.get('architecture')!r} != model {spec.architecture!r}")
    if model.get("n_layers") != spec.n_layers:
        p.append(f"plan n_layers {model.get('n_layers')} != model {spec.n_layers}")
    if [u.get("name") for u in model.get("units", [])] != [u.name for u in spec.units]:
        p.append("plan's unit list differs from the model's units")

    stages = plan.get("stages")
    if not isinstance(stages, list):
        raise PlanValidationError(p + ["stages must be a list"])

    if not plan.get("feasible"):
        if stages:
            p.append("an infeasible plan must not contain stages")
        if not (plan.get("infeasibility") or {}).get("reasons"):
            p.append("an infeasible plan must explain why (infeasibility.reasons)")
        if p:
            raise PlanValidationError(p)
        return
    if not stages:
        raise PlanValidationError(p + ["a feasible plan needs at least one stage"])

    U = len(spec.units)
    ranges = []
    for k, st in enumerate(stages):
        if st.get("stage_index") != k:
            p.append(f"stage {k} has stage_index {st.get('stage_index')} (stages must be ordered 0..{len(stages) - 1})")
        r = st.get("unit_range")
        if not (isinstance(r, list) and len(r) == 2 and all(isinstance(x, int) for x in r)):
            p.append(f"stage {k}: unit_range must be [start, end]")
            ranges.append((0, 0))
            continue
        s, e = r
        if e <= s:
            p.append(f"stage {k} ({st.get('worker_id')}): empty or reversed unit_range {r}")
        ranges.append((s, e))
    p.extend(check_ranges(ranges, U))

    # Per-stage content must match the model for that range.
    assigned = Counter()
    for k, (st, (s, e)) in enumerate(zip(stages, ranges)):
        if not (0 <= s < e <= U):
            continue
        units = spec.units[s:e]
        if st.get("units") != [u.name for u in units]:
            p.append(f"stage {k}: listed units {st.get('units')} don't match unit_range {[s, e]}")
        expected_components = [c for u in units for c in u.components]
        if st.get("components") != expected_components:
            p.append(f"stage {k}: components don't match unit_range {[s, e]} (order must be preserved)")
        blocks = [u.block_index for u in units if u.block_index is not None]
        lr = st.get("layer_range")
        expected_lr = {"first_block": blocks[0], "last_block": blocks[-1]} if blocks else None
        if lr != expected_lr:
            p.append(f"stage {k}: layer_range {lr} != {expected_lr} implied by unit_range")
        assigned.update(st.get("components") or [])
    all_components = [c.name for c in spec.components]
    for c in all_components:
        if assigned[c] == 0:
            p.append(f"model component {c} is not assigned to any stage")
        elif assigned[c] > 1:
            p.append(f"model component {c} is assigned {assigned[c]} times")
    for c in assigned:
        if c not in all_components:
            p.append(f"stage lists unknown component {c}")

    # Workers.
    stage_workers = [st.get("worker_id") for st in stages]
    for wid, n in Counter(stage_workers).items():
        if n > 1:
            p.append(f"worker {wid} holds {n} stages (each worker may hold one contiguous range)")
    unused_ids = {u.get("worker_id") for u in plan.get("unused_workers") or []}
    for wid in stage_workers:
        if wid in unused_ids:
            p.append(f"worker {wid} is both assigned a stage and listed as unused")
    if workers is not None:
        for k, st in enumerate(stages):
            w = workers.get(st.get("worker_id"))
            if w is None:
                p.append(f"stage {k}: worker {st.get('worker_id')!r} does not exist")
            elif getattr(w, "profile_source", None) != st.get("profile_source"):
                p.append(f"stage {k}: profile_source {st.get('profile_source')!r} != worker's {getattr(w, 'profile_source', None)!r}")
    has_sim = any(st.get("profile_source") == "SIMULATED" for st in stages)
    if bool(plan.get("includes_simulated_workers")) != has_sim:
        p.append("includes_simulated_workers does not match the stages' profile sources")

    # Memory.
    for k, (st, (s, e)) in enumerate(zip(stages, ranges)):
        mem, ex, avail = st.get("memory") or {}, st.get("execution") or {}, st.get("available_memory") or {}
        device = ex.get("device")
        dcap, hcap = avail.get("device_capacity_bytes"), avail.get("host_capacity_bytes")
        dbytes, hbytes = mem.get("device_bytes"), mem.get("host_bytes")
        if None in (dcap, hcap, dbytes, hbytes):
            p.append(f"stage {k}: memory requirement or capacity missing")
            continue
        if device == "cpu":
            if dbytes + hbytes > dcap:
                p.append(f"stage {k} ({st.get('worker_id')}): needs {dbytes + hbytes} B RAM > capacity {dcap} B")
        elif device == "gpu":
            if dbytes > dcap:
                p.append(f"stage {k} ({st.get('worker_id')}): needs {dbytes} B VRAM > capacity {dcap} B")
            if hbytes > hcap:
                p.append(f"stage {k} ({st.get('worker_id')}): needs {hbytes} B host RAM > capacity {hcap} B")
        else:
            p.append(f"stage {k}: unknown execution device {device!r}")
        if estimator is not None and 0 <= s < e <= U:
            actual = estimator.stage(s, e)
            if (actual.device_bytes, actual.host_bytes) != (dbytes, hbytes):
                p.append(
                    f"stage {k}: stated memory (device {dbytes}, host {hbytes}) != recomputed "
                    f"(device {actual.device_bytes}, host {actual.host_bytes}) for units {[s, e]}"
                )
        if workers is not None:
            w = workers.get(st.get("worker_id"))
            usable = getattr(w, "ram_usable_bytes", None) if w is not None else None
            if w is not None and (usable is None or hcap > usable):
                p.append(f"stage {k}: stated host capacity {hcap} exceeds the worker's usable RAM {usable}")

    if p:
        raise PlanValidationError(p)
