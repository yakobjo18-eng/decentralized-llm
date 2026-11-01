"""Command line:

  python -m partitioner inspect  --model gpt2
  python -m partitioner memory   --model gpt2 --dtype fp16 [--quant int8] [--context 512]
  python -m partitioner plan     --model gpt2 [--local computer-1] [--simulated PROFILE.yaml ...]
                                 [--orchestrator http://HOST:8080] [--out plan.json] [...]
  python -m partitioner validate PLAN.json --model gpt2 [worker sources as for plan]
  python -m partitioner execute  PLAN.json --tokens "1,2,3;4,5,6" (--local WORKER_ID | --emulate-stages-locally)
  python -m partitioner execute  PLAN.json --tokens "1,2,3" --remote --endpoint W1=HOST:PORT --endpoint W2=HOST:PORT
  python -m partitioner preflight [--worker-config worker.yaml] [--orchestrator URL] [--endpoint ID=HOST:PORT ...] [--plan PLAN.json]

Worker sources can be combined. `plan`, `validate` and `memory` only
plan. `execute` (Phase 5) runs the built-in test model
builtin:tiny-gpt2-v1 for real, in this one process; with --remote
(Phase 6) each stage runs on its assigned worker's StageExecution
service over gRPC instead.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List

from partitioner import plan as plan_mod
from partitioner.api import create_plan
from partitioner.cost import CostModel, WorkerOverride
from partitioner.inspection import inspect_model
from partitioner.memory import EstimationError, MemoryEstimator, MemoryPolicy, Precision, Workload
from partitioner.plan import fmt_bytes
from partitioner.validate import PlanValidationError, estimator_for_plan, validate_plan
from partitioner.workers import Phase3Margins, from_local_machine, from_orchestrator, from_simulated_profile


def _model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default="gpt2",
                   help="Hugging Face model id (config only is downloaded), or builtin:tiny-gpt2-v1 (offline)")
    p.add_argument("--offline", action="store_true", help="use only the local Hugging Face cache")
    p.add_argument("--no-checkpoint-check", action="store_true", help="skip reading the checkpoint's safetensors header")


def _precision_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--quant", default="none", choices=["none", "int8", "int4"])
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--context", type=int, default=None, help="KV-cache length (default: model maximum)")
    p.add_argument("--prompt", type=int, default=None, help="prefill tokens (default: --context)")
    p.add_argument("--cost-mode", default="decode", choices=["decode", "prefill"])
    p.add_argument("--logits", default="all", choices=["all", "last"])
    p.add_argument("--extra-safety", type=float, default=0.10)


def _worker_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--local", metavar="WORKER_ID", help="profile THIS machine as a REAL worker with this id")
    p.add_argument("--simulated", nargs="*", default=[], metavar="PROFILE", help="SIMULATED profile YAML files")
    p.add_argument("--orchestrator", metavar="URL", help="read registered workers from a running orchestrator")
    p.add_argument("--cpu-gflops", action="append", default=[], metavar="ID=GFLOPS")
    p.add_argument("--gpu-gflops", action="append", default=[], metavar="ID=GFLOPS")
    p.add_argument("--throughput", default="measured", choices=["measured", "estimated"])
    p.add_argument("--link-bandwidth-mbps", type=float, default=None, help="inter-stage link, megabits/s")
    p.add_argument("--no-simulated", action="store_true", help="exclude SIMULATED workers")
    p.add_argument("--require-all-workers", action="store_true", help="every eligible worker must get a stage")
    p.add_argument("--ram-margin-percent", type=float, default=Phase3Margins.ram_percent,
                   help="--local only: Phase 3 RAM safety margin, percent of total (default %(default)s)")
    p.add_argument("--ram-margin-min-gib", type=float, default=Phase3Margins.ram_min_gib,
                   help="--local only: Phase 3 RAM safety margin floor, GiB (default %(default)s)")


def _spec(args):
    from partitioner.execution.models import get_executable_model

    builtin = get_executable_model(args.model)
    if builtin is not None:
        return builtin.spec()
    return inspect_model(args.model, check_checkpoint=not args.no_checkpoint_check, local_files_only=args.offline)


def _settings(args):
    precision = Precision(weight_dtype=args.dtype, quantization=args.quant)
    workload = Workload(batch_size=args.batch, context_length=args.context, prompt_length=args.prompt,
                        cost_mode=args.cost_mode, logits_positions=args.logits)
    return precision, workload, MemoryPolicy(extra_safety_fraction=args.extra_safety)


def _workers(args, run_compute_test: bool = True):
    workers = []
    if args.orchestrator:
        workers.extend(from_orchestrator(args.orchestrator))
    for path in args.simulated:
        workers.append(from_simulated_profile(path))
    if args.local:
        print(f"profiling this machine as REAL worker {args.local!r} (backend probe ~10 s)...", file=sys.stderr)
        margins = Phase3Margins(ram_percent=args.ram_margin_percent, ram_min_gib=args.ram_margin_min_gib)
        if margins != Phase3Margins():
            print(f"note: non-default Phase 3 RAM margin for {args.local!r}: {margins.ram_percent}% / "
                  f"min {margins.ram_min_gib} GiB (orchestrator default: {Phase3Margins.ram_percent}% / "
                  f"min {Phase3Margins.ram_min_gib} GiB)", file=sys.stderr)
        workers.append(from_local_machine(args.local, margins, run_compute_test=run_compute_test))
    return workers


def _overrides(args) -> dict:
    out = {}
    for flag, field in ((args.cpu_gflops, "cpu_gflops"), (args.gpu_gflops, "gpu_gflops")):
        for item in flag:
            wid, _, val = item.partition("=")
            prev = out.get(wid, WorkerOverride())
            out[wid] = WorkerOverride(**{**prev.__dict__, field: float(val)})
    return out


def _cmd_inspect(args) -> int:
    spec = _spec(args)
    out = {"summary": spec.summary(), "config": spec.config, "inspection": spec.inspection,
           "checkpoint": spec.checkpoint.to_dict() if spec.checkpoint else None,
           "components": [{"name": c.name, "kind": c.kind, "owned_params": c.owned_numel, "tied_params": c.tied_numel}
                          for c in spec.components]}
    print(json.dumps(out, indent=2, default=str))
    return 0


def _cmd_memory(args) -> int:
    spec = _spec(args)
    precision, workload, policy = _settings(args)
    est = MemoryEstimator(spec, precision, workload, policy)
    rows = est.unit_table()
    print(f"{spec.model_id}: {precision.weight_dtype}/{precision.quantization}, batch {est.workload.batch_size}, "
          f"context {est.workload.context_length}, prompt {est.workload.prompt_length}")
    print(f"{'unit':<12}{'params':>12}{'tied':>12}{'weights':>12}{'kv cache':>12}{'act. peak':>12}  provenance")
    for r in rows:
        print(f"{r['name']:<12}{r['params']:>12,}{r['tied_params']:>12,}{fmt_bytes(r['weight_bytes']):>12}"
              f"{fmt_bytes(r['kv_cache_bytes']):>12}{fmt_bytes(r['activation_peak_bytes']):>12}  {r['weight_provenance']}")
    print("(tied = parameters shared with another unit; stored once unless the units land on different stages)")
    whole = est.stage(0, len(spec.units))
    print("\nwhole model on ONE worker:")
    for item in whole.items:
        print(f"  {item.name:<30}{fmt_bytes(item.bytes):>12}  {item.provenance.value:<10} {item.basis}")
    print(f"  {'TOTAL':<30}{fmt_bytes(whole.total_bytes):>12}  {whole.provenance.value}")
    return 0


def _cmd_plan(args) -> int:
    spec = _spec(args)
    precision, workload, policy = _settings(args)
    cm = CostModel(throughput_preference=args.throughput,
                   link_bandwidth_bytes_per_s=args.link_bandwidth_mbps * 1e6 / 8 if args.link_bandwidth_mbps else None)
    workers = _workers(args)
    if not workers:
        print("no workers given (use --local, --simulated or --orchestrator)", file=sys.stderr)
        return 2
    plan = create_plan(spec, workers, precision, workload, policy, cm, _overrides(args),
                       allow_simulated=not args.no_simulated, allow_unused_workers=not args.require_all_workers)
    if args.out:
        plan_mod.save(plan, args.out)
    _print_plan(plan)
    return 0 if plan["feasible"] else 1


def _print_plan(plan: dict) -> None:
    print(f"\nPLAN ({'FEASIBLE' if plan['feasible'] else 'INFEASIBLE'}) for {plan['model']['model_id']} -- "
          f"{plan['precision']['weight_dtype']}/{plan['precision']['quantization']}, {plan['objective']['cost_basis']}")
    if plan["includes_simulated_workers"]:
        print("!! includes SIMULATED - NOT PHYSICALLY VERIFIED workers")
    for st in plan["stages"]:
        lr = st["layer_range"]
        if not lr:
            blocks = "no blocks"
        elif lr["first_block"] == lr["last_block"]:
            blocks = f"block {lr['first_block']}"
        else:
            blocks = f"blocks {lr['first_block']}-{lr['last_block']}"
        print(f"\n stage {st['stage_index']}: {st['worker_id']} [{st['profile_source']}] units {st['unit_range']} ({blocks})")
        for line in st["explanation"]:
            print(f"   - {line}")
    if plan["feasible"]:
        print(f"\n bottleneck: {plan['objective']['bottleneck_seconds'] * 1e3:.3f} ms (ESTIMATED); "
              f"{plan['objective']['worker_order_search']}")
    for u in plan["unused_workers"]:
        print(f" unused: {u['worker_id']} [{u['profile_source']}]: {u['reason']} "
              f"(RAM available {fmt_bytes(u.get('ram_available_bytes'))}, usable {fmt_bytes(u.get('ram_usable_bytes'))})")
    for r in (plan.get("infeasibility") or {}).get("reasons", []):
        print(f" infeasible: {r}")
    for w in plan["warnings"]:
        print(f" warning: {w}")


def _cmd_validate(args) -> int:
    spec = _spec(args)
    plan = plan_mod.load(args.plan)
    workers = {w.worker_id: w for w in _workers(args)} or None
    est = None
    try:
        est = estimator_for_plan(plan, spec)
    except (KeyError, TypeError, EstimationError) as exc:
        print(f"note: memory not recomputed ({exc})", file=sys.stderr)
    try:
        validate_plan(plan, spec, workers, est)
    except PlanValidationError as exc:
        print(exc)
        return 1
    print(f"VALID: {args.plan} ({len(plan['stages'])} stages, memory {'recomputed' if est else 'not recomputed'}, "
          f"workers {'checked' if workers else 'not checked'})")
    return 0


def _parse_tokens(text: str):
    """'1,2,3;4,5,6' -> [[1, 2, 3], [4, 5, 6]] (one sequence per ';')."""
    rows = []
    for part in text.split(";"):
        part = part.strip()
        rows.append([int(t) for t in part.split(",") if t.strip()] if part else [])
    return rows


def _cmd_execute(args) -> int:
    from partitioner.execution import LOCAL_EMULATION, REAL_MODE, ExecutionLimits, execute_plan

    if [bool(args.local), args.emulate_stages_locally, args.remote].count(True) != 1:
        print("choose exactly one of --local WORKER_ID (real single-worker execution on this machine), "
              "--emulate-stages-locally (all stages in this process, labelled as emulation) or "
              "--remote (each stage on its worker over gRPC, with --endpoint ID=HOST:PORT)", file=sys.stderr)
        return 2
    endpoints = {}
    if args.endpoint and not args.remote:
        print("--endpoint is only used with --remote", file=sys.stderr)
        return 2
    for item in args.endpoint:
        from partitioner.execution.remote import parse_endpoint

        try:
            wid, addr = parse_endpoint(item)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        if wid in endpoints:
            print(f"--endpoint given twice for worker {wid!r}", file=sys.stderr)
            return 2
        endpoints[wid] = addr
    try:
        tokens = _parse_tokens(args.tokens)
    except ValueError as exc:
        print(f"--tokens must be comma-separated integers, sequences separated by ';': {exc}", file=sys.stderr)
        return 2
    mib = 2**20
    limits = ExecutionLimits(
        max_batch_size=args.max_batch,
        max_sequence_length=args.max_seq,
        max_activation_bytes=int(args.max_activation_mib * mib),
        max_weight_bytes=int(args.max_weight_mib * mib),
    )
    workers = {w.worker_id: w for w in _workers(args, run_compute_test=False)} or None
    if args.remote:
        from partitioner.execution.remote import RemoteOptions, execute_plan_remote

        options = RemoteOptions(stage_timeout_seconds=args.stage_timeout, connect_timeout_seconds=args.connect_timeout,
                                max_attempts=args.max_attempts, max_tensor_bytes=int(args.max_activation_mib * mib),
                                max_message_bytes=int(args.max_activation_mib * mib) * 2 + 4 * mib)
        result = execute_plan_remote(args.plan, tokens, endpoints, workers=workers, limits=limits, options=options,
                                     check_reference=not args.no_reference_check, include_logits=args.include_logits)
    else:
        result = execute_plan(
            args.plan, tokens,
            mode=REAL_MODE if args.local else LOCAL_EMULATION,
            workers=workers, local_worker_id=args.local, limits=limits,
            check_reference=not args.no_reference_check, include_logits=args.include_logits,
        )
    text = json.dumps(result, indent=2, default=str)
    if args.out:
        from pathlib import Path

        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    err = result["error"]
    summary = f"{result['status'].upper()} ({result['execution_mode']}, distributed: {str(result['distributed']).lower()})"
    if err:
        detail = err["problems"][0] if err["problems"] else f"{err['type']}: {err['message'].splitlines()[0]}"
        more = f" (+{len(err['problems']) - 1} more)" if len(err["problems"]) > 1 else ""
        summary += f": {err['phase']}: {detail}{more}"
    print(summary, file=sys.stderr)
    return 0 if result["status"] == "succeeded" else 1


def _cmd_preflight(args) -> int:
    from partitioner import preflight
    from partitioner.execution.remote import parse_endpoint

    endpoints = {}
    for item in args.endpoint:
        try:
            wid, addr = parse_endpoint(item)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        endpoints[wid] = addr
    report = preflight.run(args.worker_config, args.orchestrator, args.expect_worker, endpoints, args.plan,
                           args.require_multi_host, args.timeout, args.skip_runtime,
                           orchestrator_configs=args.orchestrator_config, probe_orchestrator=args.probe_orchestrator,
                           peer_ip=args.peer_ip)
    if args.out:
        from pathlib import Path

        Path(args.out).write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    print(json.dumps(report.to_dict(), indent=2) if args.json else report.render())
    return 1 if report.failed else 0


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m partitioner", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("inspect", help="inspect a model's architecture (no weights)")
    _model_args(p)
    p = sub.add_parser("memory", help="per-unit and whole-model memory estimate")
    _model_args(p)
    _precision_args(p)
    p = sub.add_parser("plan", help="create a partition plan")
    _model_args(p)
    _precision_args(p)
    _worker_args(p)
    p.add_argument("--out", help="write the plan JSON here")
    p = sub.add_parser("validate", help="validate a saved plan")
    p.add_argument("plan")
    _model_args(p)
    _worker_args(p)
    p = sub.add_parser("execute", help="execute a validated plan (Phase 5: builtin:tiny-gpt2-v1 only, one process)")
    p.add_argument("plan")
    p.add_argument("--tokens", required=True, help='token ids, e.g. "1,2,3" or "1,2,3;4,5,6" for a batch of two')
    p.add_argument("--emulate-stages-locally", action="store_true",
                   help="run EVERY stage in this process regardless of its assigned worker (labelled; not distributed)")
    p.add_argument("--remote", action="store_true",
                   help="Phase 6: run each stage on its assigned worker's StageExecution service over gRPC")
    p.add_argument("--endpoint", action="append", default=[], metavar="WORKER_ID=HOST:PORT",
                   help="--remote: a worker's StageExecution address (one per stage worker)")
    p.add_argument("--stage-timeout", type=float, default=30.0, help="--remote: gRPC deadline per stage, seconds")
    p.add_argument("--connect-timeout", type=float, default=3.0, help="--remote: connection wait per attempt, seconds")
    p.add_argument("--max-attempts", type=int, default=3,
                   help="--remote: attempts per stage, retried only when the request was not executed")
    _worker_args(p)
    p.add_argument("--max-batch", type=int, default=8)
    p.add_argument("--max-seq", type=int, default=None, help="default: the model's maximum positions")
    p.add_argument("--max-activation-mib", type=float, default=64.0)
    p.add_argument("--max-weight-mib", type=float, default=64.0)
    p.add_argument("--no-reference-check", action="store_true", help="skip the unpartitioned reference comparison")
    p.add_argument("--include-logits", action="store_true", help="include the full logits in the result")
    p.add_argument("--out", help="also write the result JSON here")
    p = sub.add_parser("preflight", help="Phase 7: deployment diagnostics (config, reachability, protocol, plan, limits)")
    p.add_argument("--worker-config", action="append", default=[], metavar="YAML", help="check a worker config on THIS machine")
    p.add_argument("--orchestrator", metavar="URL", help="orchestrator REST URL, e.g. http://192.168.1.10:8080")
    p.add_argument("--expect-worker", action="append", default=[], metavar="ID", help="worker that must be registered")
    p.add_argument("--endpoint", action="append", default=[], metavar="WORKER_ID=HOST:PORT")
    p.add_argument("--plan", help="plan JSON to check against the endpoints and the orchestrator")
    p.add_argument("--require-multi-host", action="store_true", help="FAIL unless the plan spans two or more hosts")
    p.add_argument("--timeout", type=float, default=3.0, help="per network check, seconds")
    p.add_argument("--skip-runtime", action="store_true", help="skip the dependency checks")
    p.add_argument("--orchestrator-config", action="append", default=[], metavar="YAML",
                   help="Phase 8: check the orchestrator's config on THIS machine (bind, ownership, port, firewall)")
    p.add_argument("--probe-orchestrator", action="store_true",
                   help="Phase 8: TCP-connect to each --worker-config's orchestrator_address (start the orchestrator first)")
    p.add_argument("--peer-ip", default="", metavar="IP",
                   help="the other machine's IP, used only to print exact firewall commands in hints")
    p.add_argument("--json", action="store_true", help="print the report as JSON")
    p.add_argument("--out", help="also write the JSON report here")
    args = parser.parse_args(argv)
    return {"inspect": _cmd_inspect, "memory": _cmd_memory, "plan": _cmd_plan, "validate": _cmd_validate,
            "execute": _cmd_execute, "preflight": _cmd_preflight}[args.cmd](args)
