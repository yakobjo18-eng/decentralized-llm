"""Phase 6 coordinator: run a validated plan with each stage on its
assigned worker's StageExecution service, over gRPC.

    result = execute_plan_remote("plan.json", [[1, 2, 3]],
                                 endpoints={"w-a": "192.168.1.20:50200", "w-b": "192.168.1.21:50200"})

The coordinator validates the plan (Phase 5 loader, remote mode), checks
the input against the limits before allocating anything, then for each
stage in order sends ExecuteStageRequest(plan, stage, activation) to the
stage's worker and passes the returned activation -- as protobuf bytes
over gRPC, never shared memory -- to the next stage's worker.

Retry policy (bounded, max_attempts): a stage request is retried ONLY
when it is known not to have executed:
  - the channel did not become ready within connect_timeout_seconds
    (the request was never sent);
  - the worker answered BUSY (refused before executing);
  - gRPC RESOURCE_EXHAUSTED (the server refused the call before the
    handler ran).
Anything else -- deadline exceeded, connection lost mid-call, worker
FAILED or DEADLINE_EXCEEDED -- is reported and NOT retried, because the
stage may have run. (Stages are stateless today, so a replay would be
harmless; the policy is set for stateful stages, e.g. a KV cache.)

Never raises for a bad plan, input, worker or network: the result has
status "rejected", "failed" or "succeeded", as in Phase 5.
"""

from __future__ import annotations

import dataclasses
import json
import os
import socket
import time
import uuid
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from partitioner.execution import wire
from partitioner.execution.interface import INPUT_STAGE, STAGE_INTERFACE_VERSION, TOKEN_IDS, Activation, ActivationMeta
from partitioner.execution.limits import ExecutionLimits, check_token_ids, preflight
from partitioner.execution.loader import REMOTE_MODE, ExecutionRejected, load_for_execution
from partitioner.execution.runner import RESULT_SCHEMA_VERSION, BoundaryError, _check_boundary, _error

MIB = 2**20
# Remote logits vs. the coordinator's local unpartitioned reference.
# Same machine and kernels: bit-identical (raw fp32 bytes, no rounding in
# transit). Different CPUs / BLAS builds may order fp32 sums differently;
# 1e-4 absolute on logits of magnitude ~11 allows for that.
REMOTE_REFERENCE_ATOL = 1e-4
TRANSPORT = "gRPC StageExecution protocol v1 (insecure channel: unauthenticated, unencrypted)"


@dataclasses.dataclass(frozen=True)
class RemoteOptions:
    stage_timeout_seconds: float = 30.0  # gRPC deadline per stage request
    connect_timeout_seconds: float = 3.0
    max_attempts: int = 3
    retry_backoff_seconds: float = 0.25
    max_message_bytes: int = 32 * MIB
    max_tensor_bytes: int = 16 * MIB

    def validate(self) -> List[str]:
        p = []
        if self.stage_timeout_seconds <= 0 or self.connect_timeout_seconds <= 0:
            p.append("timeouts must be positive")
        if not 1 <= self.max_attempts <= 10:
            p.append("max_attempts must be in 1..10")
        if self.retry_backoff_seconds < 0:
            p.append("retry_backoff_seconds must not be negative")
        if self.max_tensor_bytes <= 0 or self.max_message_bytes <= self.max_tensor_bytes:
            p.append("max_message_bytes must exceed max_tensor_bytes, both positive")
        return p


class RemoteStageError(RuntimeError):
    def __init__(self, code: str, message: str, problems=(), executed: Optional[bool] = None, attempts=()):
        super().__init__(message)
        self.code, self.problems, self.executed, self.attempts = code, list(problems), executed, list(attempts)


def parse_endpoint(text: str) -> Tuple[str, str]:
    """'worker-id=host:port' -> ('worker-id', 'host:port')."""
    wid, sep, addr = text.partition("=")
    host, colon, port = addr.rpartition(":")
    if not sep or not wid or not colon or not host or not port.isdigit() or not 0 < int(port) < 65536:
        raise ValueError(f"endpoint {text!r} must look like WORKER_ID=HOST:PORT")
    return wid, addr


def _grpc():
    import grpc

    from partitioner.workers import _ensure_worker_importable

    _ensure_worker_importable()
    from worker.generated import stage_execution_pb2 as pb
    from worker.generated import stage_execution_pb2_grpc as pb_grpc

    return grpc, pb, pb_grpc


def _dispatch(grpc, pb, stub, channel, request, options: RemoteOptions) -> Tuple[object, List[dict]]:
    attempts: List[dict] = []
    for n in range(1, options.max_attempts + 1):
        last = n == options.max_attempts
        ready = grpc.channel_ready_future(channel)
        try:
            ready.result(timeout=options.connect_timeout_seconds)
        except grpc.FutureTimeoutError:
            ready.cancel()
            attempts.append({"attempt": n, "outcome": "unreachable", "executed": False, "retried": not last,
                             "detail": f"no connection within {options.connect_timeout_seconds}s; request not sent"})
            if not last:
                time.sleep(options.retry_backoff_seconds * n)
            continue
        t = time.perf_counter()
        try:
            response = stub.ExecuteStage(request, timeout=options.stage_timeout_seconds)
        except grpc.RpcError as exc:
            code = exc.code()
            if code == grpc.StatusCode.RESOURCE_EXHAUSTED:
                attempts.append({"attempt": n, "outcome": "grpc RESOURCE_EXHAUSTED", "executed": False,
                                 "retried": not last, "detail": exc.details()})
                if not last:
                    time.sleep(options.retry_backoff_seconds * n)
                continue
            attempts.append({"attempt": n, "outcome": f"grpc {code.name}", "executed": None, "retried": False,
                             "detail": exc.details(), "seconds": time.perf_counter() - t})
            raise RemoteStageError(
                "AMBIGUOUS_FAILURE",
                f"gRPC {code.name} ({exc.details()}): the stage may or may not have executed, so it was not retried",
                executed=None, attempts=attempts,
            ) from exc
        status = pb.StageStatus.Name(response.status)
        if response.status == pb.STAGE_STATUS_BUSY:
            attempts.append({"attempt": n, "outcome": status, "executed": False, "retried": not last,
                             "detail": response.error.message})
            if not last:
                time.sleep(options.retry_backoff_seconds * n)
            continue
        attempts.append({"attempt": n, "outcome": status, "executed": response.error.executed if response.error.code else True,
                         "retried": False, "seconds": time.perf_counter() - t})
        if response.status != pb.STAGE_STATUS_OK:
            raise RemoteStageError(response.error.code or status, response.error.message or status,
                                   problems=response.error.problems, executed=response.error.executed, attempts=attempts)
        return response, attempts
    raise RemoteStageError("RETRIES_EXHAUSTED",
                           f"not executed after {options.max_attempts} attempts (last: {attempts[-1]['outcome']})",
                           executed=False, attempts=attempts)


def execute_plan_remote(
    source,
    token_ids: Sequence[Sequence[int]],
    endpoints: Mapping[str, str],
    workers: Optional[Mapping[str, object]] = None,
    limits: ExecutionLimits = ExecutionLimits(),
    options: RemoteOptions = RemoteOptions(),
    request_id: Optional[str] = None,
    check_reference: bool = True,
    include_logits: bool = False,
) -> dict:
    t0 = time.perf_counter()
    host = socket.gethostname()
    request_id = request_id or uuid.uuid4().hex
    result = {
        "execution_result_schema_version": RESULT_SCHEMA_VERSION,
        "status": None,
        "execution_mode": REMOTE_MODE,
        "distributed": False,
        "distributed_note": None,
        "request_id": request_id,
        "stage_interface_version": STAGE_INTERFACE_VERSION,
        "protocol_version": wire.PROTOCOL_VERSION,
        "transport": TRANSPORT,
        "coordinator": {"hostname": host, "pid": os.getpid()},
        "model": None,
        "plan": None,
        "worker_assignments": [],
        "limits": limits.to_dict(),
        "remote_options": dataclasses.asdict(options),
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
        result["elapsed"] = {
            "total_seconds": time.perf_counter() - t0,
            "provenance": "MEASURED: wall clock (time.perf_counter) of this one run on the coordinator; not a benchmark",
        }
        procs = {(s["worker_reported"]["hostname"], s["worker_reported"]["pid"]) for s in result["stages"]}
        hosts = {h for h, _ in procs}
        result["distinct_worker_processes"] = len(procs)
        result["distinct_worker_hostnames"] = sorted(hosts)
        result["distributed"] = len(procs) > 1
        result["distributed_note"] = (
            f"{len(result['stages'])} stage(s) ran in {len(procs)} separate worker process(es) on {len(hosts)} host(s), "
            "activations passed over gRPC. Hostnames and pids are reported by the workers (unauthenticated); "
            "one host means this was NOT a multi-machine run."
        )
        # Phase 7 test classification (docs/DEPLOYMENT_LAN.md). Workers in
        # this coordinator's own process count as one process.
        if not procs:
            result["test_classification"] = None
        elif len(hosts) > 1:
            result["test_classification"] = "REAL_MULTI_MACHINE"
        elif len(procs) > 1 or any(pid != os.getpid() for _, pid in procs):
            result["test_classification"] = "REAL_MULTI_PROCESS"
        else:
            result["test_classification"] = "REAL_SINGLE_MACHINE"
        result["test_classification_basis"] = (
            "derived from the hostnames/pids the workers reported (unauthenticated); REAL_MULTI_MACHINE needs two or more "
            "distinct hostnames" if procs else "no stage ran"
        )
        return result

    def reject(problems, phase):
        result["error"] = _error(ExecutionRejected(problems, phase=phase), phase)
        return finish("rejected")

    problems = options.validate()
    if problems:
        return reject(problems, "remote_options")
    try:
        loaded = load_for_execution(source, REMOTE_MODE, workers, None, host)
    except ExecutionRejected as exc:
        result["plan"] = exc.plan_info
        result["error"] = _error(exc, exc.phase)
        return finish("rejected")
    plan, spec, model = loaded.plan, loaded.spec, loaded.model
    result["plan"] = loaded.plan_info
    result["warnings"].extend(loaded.warnings)
    result["model"] = {"model_id": spec.model_id, "description": model.description,
                       "config_fingerprint": spec.config_fingerprint, "weights_version": model.weights_version,
                       "total_params": spec.total_params, "units": [u.name for u in spec.units]}
    problems = []
    for st in plan["stages"]:
        addr = endpoints.get(st["worker_id"])
        if addr is None:
            problems.append(f"no endpoint given for worker {st['worker_id']!r} (stage {st['stage_index']})")
        else:
            try:
                parse_endpoint(f"{st['worker_id']}={addr}")
            except ValueError as exc:
                problems.append(str(exc))
        result["worker_assignments"].append({
            "stage_index": st["stage_index"], "worker_id": st["worker_id"], "profile_source": st["profile_source"],
            "planned_hostname": st.get("hostname"), "unit_range": st["unit_range"], "units": st["units"],
            "endpoint": addr,
        })
    if problems:
        return reject(problems, "endpoints")

    problems = check_token_ids(token_ids, spec.vocab_size)
    if problems:
        return reject(problems, "input_validation")
    batch, tokens = len(token_ids), len(token_ids[0])
    result["input"] = {"kind": TOKEN_IDS, "shape": [batch, tokens], "dtype": "int64"}
    ranges = [ss.unit_range for ss in loaded.stage_specs]
    ref_bytes = model.weight_bytes(spec, 0, len(spec.units)) if check_reference else 0
    pf, problems = preflight(spec, batch, tokens, ranges, plan["precision"]["activation_dtype"], ref_bytes, limits,
                             plan_batch=plan["workload"].get("batch_size"), plan_prompt=plan["workload"].get("prompt_length"))
    if pf is not None and pf.largest_boundary_bytes > options.max_tensor_bytes:
        problems.append(f"largest boundary tensor {pf.largest_boundary_bytes} B > max_tensor_bytes {options.max_tensor_bytes}")
    if problems:
        return reject(problems, "resource_limits")
    result["preflight"] = pf.to_dict()

    grpc, pb, pb_grpc = _grpc()
    channel_options = [("grpc.max_receive_message_length", options.max_message_bytes),
                       ("grpc.max_send_message_length", options.max_message_bytes)]
    channels: Dict[str, object] = {}
    plan_json = json.dumps(plan).encode("utf-8")
    phase, stage_index = "dispatch", None
    try:
        x_tensor = model.token_tensor(token_ids)
        payload = wire.encode(Activation(
            ActivationMeta(STAGE_INTERFACE_VERSION, request_id, TOKEN_IDS, (batch, tokens), "int64", INPUT_STAGE), x_tensor))
        out = None
        for ss, st in zip(loaded.stage_specs, plan["stages"]):
            stage_index = ss.stage_index
            addr = endpoints[st["worker_id"]]
            if addr not in channels:
                channels[addr] = grpc.insecure_channel(addr, options=channel_options)
            channel = channels[addr]
            request = pb.ExecuteStageRequest(
                protocol_version=pb.PROTOCOL_VERSION_1, request_id=request_id, plan_id=plan["inputs_fingerprint"],
                plan_json=plan_json, model_id=spec.model_id, config_fingerprint=spec.config_fingerprint,
                target_worker_id=st["worker_id"], stage_index=ss.stage_index, unit_start=ss.unit_range[0],
                unit_end=ss.unit_range[1], input=payload, coordinator=f"{host}/{os.getpid()}",
            )
            ts = time.perf_counter()
            entry = {"stage_index": ss.stage_index, "worker_id": st["worker_id"], "endpoint": addr,
                     "unit_range": list(ss.unit_range), "request_bytes": request.ByteSize()}
            try:
                response, attempts = _dispatch(grpc, pb, pb_grpc.StageExecutionStub(channel), channel, request, options)
            except RemoteStageError as exc:
                entry["attempts"] = exc.attempts
                result["failed_stage"] = entry
                raise
            round_trip = time.perf_counter() - ts
            problems = []
            if response.protocol_version != pb.PROTOCOL_VERSION_1:
                problems.append(f"response protocol_version {int(response.protocol_version)} != 1")
            if response.request_id != request_id:
                problems.append(f"response request_id {response.request_id!r} != {request_id!r}")
            if response.worker_id != st["worker_id"]:
                problems.append(f"response came from worker {response.worker_id!r}, expected {st['worker_id']!r}")
            if response.stage_index != ss.stage_index:
                problems.append(f"response stage_index {response.stage_index} != {ss.stage_index}")
            if not response.HasField("output"):
                problems.append("OK response without an output tensor")
            if problems:
                raise BoundaryError(f"stage {ss.stage_index} response rejected: " + "; ".join(problems))
            try:
                out = wire.decode(response.output, options.max_tensor_bytes, request_id)
            except wire.WireError as exc:
                raise BoundaryError(f"stage {ss.stage_index} output tensor rejected: {exc}") from exc
            _check_boundary(out, ss, batch, tokens, spec.hidden_size, spec.vocab_size, request_id)
            in_kind, in_dtype, in_shape = wire.describe(payload)
            entry.update({
                "worker_reported": {"hostname": response.worker_hostname, "pid": response.worker_pid},
                "input": {"kind": in_kind, "shape": list(in_shape), "dtype": in_dtype},
                "output": {"kind": out.meta.kind, "shape": list(out.meta.shape), "dtype": out.meta.dtype},
                "response_bytes": response.ByteSize(),
                "attempts": attempts,
                "round_trip_seconds": round_trip,
                "worker_timing": {"validate_seconds": response.timing.validate_seconds,
                                  "compute_seconds": response.timing.compute_seconds,
                                  "total_seconds": response.timing.total_seconds},
            })
            result["stages"].append(entry)
            payload = response.output  # forwarded as received (already validated)
        stage_index = None
        logits = out.tensor
        result["output"] = {"kind": out.meta.kind, "shape": list(logits.shape), "dtype": out.meta.dtype,
                            "next_token_ids": logits[:, -1, :].argmax(dim=-1).tolist(),
                            "logits_min": float(logits.min()), "logits_max": float(logits.max())}
        if include_logits:
            result["output"]["logits"] = logits.tolist()
        if check_reference:
            phase = "reference_check"
            ref = model.reference_forward(x_tensor)
            diff = float((ref - logits).abs().max())
            result["reference_check"] = {
                "compared_to": "Phase 5 unpartitioned local reference forward pass on the coordinator",
                "max_abs_diff": diff, "atol": REMOTE_REFERENCE_ATOL, "passed": diff <= REMOTE_REFERENCE_ATOL,
            }
            if diff > REMOTE_REFERENCE_ATOL:
                raise BoundaryError(f"remote output differs from the local reference by {diff} > {REMOTE_REFERENCE_ATOL}")
    except RemoteStageError as exc:
        err = _error(exc, phase, stage_index)
        err.update(code=exc.code, problems=exc.problems, executed=exc.executed, attempts=exc.attempts)
        result["error"] = err
        return finish("failed")
    except Exception as exc:
        result["error"] = _error(exc, phase, stage_index)
        return finish("failed")
    finally:
        for ch in channels.values():
            ch.close()
    return finish("succeeded")
