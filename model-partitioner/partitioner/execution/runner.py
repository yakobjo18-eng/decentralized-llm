"""Execute a validated plan and return a structured result.

    result = execute_plan("plan.json", [[1, 2, 3]], mode=REAL_MODE,
                          workers={"computer-1": record}, local_worker_id="computer-1")

Never raises for a bad plan, a bad input or a failing stage: the result
has status "rejected" (nothing was allocated or run), "failed" (an
error during execution; type, message, cause chain and traceback
included) or "succeeded". Only programming errors in the call itself
(e.g. an unknown mode) raise.

Every stage runs in THIS process and activations move through an
in-process transport. `distributed` is always false in Phase 5.
"""

from __future__ import annotations

import os
import socket
import time
import traceback
import uuid
from typing import Callable, Mapping, Optional, Sequence

from partitioner.execution.dtypes import torch_dtype_name
from partitioner.execution.interface import (
    INPUT_STAGE,
    STAGE_INTERFACE_VERSION,
    TOKEN_IDS,
    Activation,
    ActivationMeta,
    InProcessTransport,
    InterfaceError,
    expected_output,
)
from partitioner.execution.limits import ExecutionLimits, check_token_ids, preflight
from partitioner.execution.loader import LOCAL_EMULATION, REAL_MODE, ExecutionRejected, load_for_execution

RESULT_SCHEMA_VERSION = 1
# Partitioned vs. unpartitioned reference, fp32, same kernels and
# shapes: expected to be bit-identical; this is the documented bound.
REFERENCE_ATOL = 1e-6


class BoundaryError(RuntimeError):
    """A stage produced an activation that doesn't match what the next
    stage (or the final output) requires."""


def _error(exc: BaseException, phase: str, stage_index: Optional[int] = None) -> dict:
    causes = []
    seen = set()
    cur = exc.__cause__ or exc.__context__
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        causes.append({"type": type(cur).__name__, "message": str(cur)})
        cur = cur.__cause__ or cur.__context__
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "phase": phase,
        "stage_index": stage_index,
        "problems": list(getattr(exc, "problems", []) or []),
        "causes": causes,
        "traceback": traceback.format_exception(type(exc), exc, exc.__traceback__)[-12:],
    }


def _check_boundary(out: Activation, stage_spec, batch: int, tokens: int, hidden: int, vocab: int, request_id: str) -> None:
    """The runner's own check of a stage's output -- independent of the
    metadata the stage wrote about itself."""
    import torch

    if not isinstance(out, Activation):
        raise BoundaryError(f"stage {stage_spec.stage_index} returned {type(out).__name__}, not an Activation")
    problems = []
    want_shape = expected_output(stage_spec.output_kind, batch, tokens, hidden, vocab)
    actual_shape = tuple(out.tensor.shape)
    actual_dtype = torch_dtype_name(out.tensor.dtype)
    if out.meta.kind != stage_spec.output_kind:
        problems.append(f"kind {out.meta.kind!r} != expected {stage_spec.output_kind!r}")
    if actual_shape != want_shape:
        problems.append(f"shape {list(actual_shape)} != expected {list(want_shape)}")
    if actual_dtype != stage_spec.activation_dtype:
        problems.append(f"dtype {actual_dtype} != expected {stage_spec.activation_dtype}")
    if out.meta.produced_by_stage != stage_spec.stage_index:
        problems.append(f"produced_by_stage {out.meta.produced_by_stage} != {stage_spec.stage_index}")
    if out.meta.request_id != request_id:
        problems.append(f"request_id {out.meta.request_id!r} != {request_id!r}")
    if (tuple(out.meta.shape), out.meta.dtype) != (actual_shape, actual_dtype):
        problems.append(f"metadata ({list(out.meta.shape)}, {out.meta.dtype}) doesn't describe the tensor "
                        f"({list(actual_shape)}, {actual_dtype})")
    if not problems and not bool(torch.isfinite(out.tensor).all()):
        problems.append("output contains NaN or infinite values")
    if problems:
        raise BoundaryError(f"stage {stage_spec.stage_index} output rejected: " + "; ".join(problems))


def execute_plan(
    source,
    token_ids: Sequence[Sequence[int]],
    mode: str = REAL_MODE,
    workers: Optional[Mapping[str, object]] = None,
    local_worker_id: Optional[str] = None,
    local_hostname: Optional[str] = None,
    limits: ExecutionLimits = ExecutionLimits(),
    request_id: Optional[str] = None,
    check_reference: bool = True,
    include_logits: bool = False,
    stage_factory: Optional[Callable] = None,
) -> dict:
    if mode not in (REAL_MODE, LOCAL_EMULATION):
        raise ValueError(f"mode must be {REAL_MODE!r} or {LOCAL_EMULATION!r}, got {mode!r} "
                         "(remote execution: partitioner.execution.remote.execute_plan_remote)")
    t0 = time.perf_counter()
    host = local_hostname or socket.gethostname()
    request_id = request_id or uuid.uuid4().hex
    transport = InProcessTransport()
    result = {
        "execution_result_schema_version": RESULT_SCHEMA_VERSION,
        "status": None,
        "execution_mode": "local-single-worker" if mode == REAL_MODE else LOCAL_EMULATION,
        "distributed": False,
        "distributed_note": "Phase 5: every stage runs in this one process; no activation crosses a process or machine boundary.",
        "request_id": request_id,
        "stage_interface_version": STAGE_INTERFACE_VERSION,
        "transport": transport.name,
        "executed_on": {"hostname": host, "pid": os.getpid(), "worker_id": local_worker_id},
        "model": None,
        "plan": None,
        "worker_assignments": [],
        "limits": limits.to_dict(),
        "input": None,
        "preflight": None,
        "stages": [],
        "output": None,
        "reference_check": None,
        "elapsed": None,
        "warnings": [],
        "error": None,
    }

    def finish(status: str) -> dict:
        result["status"] = status
        # Phase 7 test classification: everything here runs in one process.
        simulated = bool((result["plan"] or {}).get("includes_simulated_workers"))
        result["test_classification"] = "SIMULATED" if simulated else "REAL_SINGLE_MACHINE"
        result["elapsed"] = {
            "total_seconds": time.perf_counter() - t0,
            "provenance": "MEASURED: wall clock (time.perf_counter) of this one run in this process; not a benchmark",
        }
        return result

    # ---- plan ------------------------------------------------------
    try:
        loaded = load_for_execution(source, mode, workers, local_worker_id, host)
    except ExecutionRejected as exc:
        result["plan"] = exc.plan_info
        result["error"] = _error(exc, exc.phase)
        return finish("rejected")
    plan, spec, model = loaded.plan, loaded.spec, loaded.model
    result["plan"] = loaded.plan_info
    result["warnings"].extend(loaded.warnings)
    result["model"] = {
        "model_id": spec.model_id,
        "description": model.description,
        "config_fingerprint": spec.config_fingerprint,
        "weights_version": model.weights_version,
        "total_params": spec.total_params,
        "units": [u.name for u in spec.units],
    }
    for st in plan["stages"]:
        result["worker_assignments"].append({
            "stage_index": st["stage_index"],
            "worker_id": st["worker_id"],
            "profile_source": st["profile_source"],
            "planned_hostname": st.get("hostname"),
            "unit_range": st["unit_range"],
            "units": st["units"],
            "executed_on_hostname": host,
            "assignment_honored": mode == REAL_MODE,
        })

    # ---- input and limits: nothing allocated yet --------------------
    problems = check_token_ids(token_ids, spec.vocab_size)
    if problems:
        result["error"] = _error(ExecutionRejected(problems, phase="input_validation"), "input_validation")
        return finish("rejected")
    batch, tokens = len(token_ids), len(token_ids[0])
    result["input"] = {"kind": TOKEN_IDS, "shape": [batch, tokens], "dtype": "int64"}
    ranges = [ss.unit_range for ss in loaded.stage_specs]
    weight_bytes = sum(model.weight_bytes(spec, s, e) for s, e in ranges)
    if check_reference:
        weight_bytes += model.weight_bytes(spec, 0, len(spec.units))
    wl = plan["workload"]
    pf, problems = preflight(spec, batch, tokens, ranges, plan["precision"]["activation_dtype"], weight_bytes, limits,
                             plan_batch=wl.get("batch_size"), plan_prompt=wl.get("prompt_length"))
    if problems:
        result["error"] = _error(ExecutionRejected(problems, phase="resource_limits"), "resource_limits")
        return finish("rejected")
    result["preflight"] = pf.to_dict()

    # ---- execution -------------------------------------------------
    factory = stage_factory or model.build_stage
    phase, stage_index = "stage_construction", None
    try:
        stages = []
        for ss in loaded.stage_specs:
            stage_index = ss.stage_index
            stages.append(factory(ss, spec))
        phase = "execution"
        x_tensor = model.token_tensor(token_ids)
        x = Activation(ActivationMeta(STAGE_INTERFACE_VERSION, request_id, TOKEN_IDS, (batch, tokens), "int64", INPUT_STAGE), x_tensor)
        for k, stage in enumerate(stages):
            stage_index = k
            ss = loaded.stage_specs[k]
            ts = time.perf_counter()
            out = stage.forward(x)
            elapsed = time.perf_counter() - ts
            _check_boundary(out, ss, batch, tokens, spec.hidden_size, spec.vocab_size, request_id)
            result["stages"].append({
                "stage_index": k,
                "worker_id": plan["stages"][k]["worker_id"],
                "unit_range": list(ss.unit_range),
                "input": {"kind": x.meta.kind, "shape": list(x.meta.shape), "dtype": x.meta.dtype},
                "output": {"kind": out.meta.kind, "shape": list(out.meta.shape), "dtype": out.meta.dtype},
                "elapsed_seconds": elapsed,
            })
            x = transport.send(out, k + 1) if k + 1 < len(stages) else out
        stage_index = None
        logits = x.tensor
        result["output"] = {
            "kind": x.meta.kind,
            "shape": list(logits.shape),
            "dtype": x.meta.dtype,
            "next_token_ids": logits[:, -1, :].argmax(dim=-1).tolist(),
            "logits_min": float(logits.min()),
            "logits_max": float(logits.max()),
        }
        if include_logits:
            result["output"]["logits"] = logits.tolist()
        result["handoffs"] = [
            {"from_stage": h.from_stage, "to_stage": h.to_stage, "kind": h.kind, "shape": list(h.shape),
             "dtype": h.dtype, "nbytes": h.nbytes, "transport": h.transport}
            for h in transport.handoffs
        ]

        if check_reference:
            phase = "reference_check"
            ref = model.reference_forward(x_tensor)
            diff = float((ref - logits).abs().max())
            passed = diff <= REFERENCE_ATOL
            result["reference_check"] = {
                "compared_to": "unpartitioned reference forward pass (all units, one weight set, no stages/transport)",
                "max_abs_diff": diff,
                "atol": REFERENCE_ATOL,
                "passed": passed,
            }
            if not passed:
                raise BoundaryError(f"partitioned output differs from the unpartitioned reference by {diff} > {REFERENCE_ATOL}")
    except Exception as exc:  # reported, not hidden: type, message, causes, traceback
        result["error"] = _error(exc, phase, stage_index)
        return finish("failed")
    return finish("succeeded")


__all__ = ["BoundaryError", "InterfaceError", "LOCAL_EMULATION", "REAL_MODE", "REFERENCE_ATOL", "execute_plan"]
