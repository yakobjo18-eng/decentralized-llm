"""Phase 6 StageExecution service: run ONE assigned stage of a validated
plan on a remote activation.

Request handling, in order (nothing is allocated or executed until
every check before it has passed):
  1. concurrency slot (else BUSY, not executed -- safe to retry);
  2. protocol version, required fields, target worker id;
  3. deadline: the caller must set one, with at least
     min_deadline_seconds left;
  4. plan: plan_json must hash to plan_id; validated once per plan id
     (Phase 5 loader in remote mode: fingerprint, feasibility,
     executable model, fp32, validate_plan with memory recomputed, no
     SIMULATED workers), then cached;
  5. assignment: plan stage `stage_index` must be assigned to THIS
     worker id on THIS host, REAL, with the requested unit range,
     model id and config fingerprint;
  6. input tensor metadata vs. the stage (kind, dtype, rank, shape,
     producer, byte length) and limits (tensor bytes, batch, sequence,
     plan workload, output bytes) -- still nothing allocated;
  7. decode (raw-le-c-v1, no pickle) + finite check;
  8. execute the stage (weights for its components only, cached);
  9. deadline re-check (late results are discarded, DEADLINE_EXCEEDED);
 10. encode the output.

Logs carry ids, shapes, status and timings -- never tensor values,
payload bytes or the plan body.

UNAUTHENTICATED: the worker id and plan in a request are claims. The
assignment check stops misrouted requests, not a malicious client on
the network (docs/REMOTE_EXECUTION.md, "Security").
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import socket
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Optional

from worker.generated import stage_execution_pb2 as pb
from worker.generated import stage_execution_pb2_grpc as pb_grpc

logger = logging.getLogger("worker.execution")

PROTOCOL_VERSION = pb.PROTOCOL_VERSION_1
MIB = 2**20


def ensure_partitioner_importable() -> None:
    """The stage implementation, plan validator and tensor codec live in
    model-partitioner/ (mirror of partitioner.workers._ensure_worker_importable)."""
    root = Path(__file__).resolve().parents[3] / "model-partitioner"
    if str(root) not in sys.path and root.is_dir():
        sys.path.insert(0, str(root))


@dataclasses.dataclass(frozen=True)
class ServiceConfig:
    worker_id: str
    hostname: str = dataclasses.field(default_factory=socket.gethostname)
    max_tensor_bytes: int = 16 * MIB
    max_message_bytes: int = 32 * MIB
    max_concurrent: int = 2
    max_batch: int = 8
    max_sequence: int = 0  # 0 = model maximum
    min_deadline_seconds: float = 0.05
    plan_cache_size: int = 8
    stage_cache_size: int = 16


class _Stop(Exception):
    """Ends request handling with a non-OK response."""

    def __init__(self, status, code: str, message: str, problems=(), executed: bool = False):
        super().__init__(message)
        self.status, self.code, self.problems, self.executed = status, code, list(problems), executed


def _reject(code: str, message: str, problems=()) -> _Stop:
    return _Stop(pb.STAGE_STATUS_REJECTED, code, message, problems)


class StageExecutionServicer(pb_grpc.StageExecutionServicer):
    def __init__(self, config: ServiceConfig, stage_factory: Optional[Callable] = None,
                 before_compute: Optional[Callable] = None):
        ensure_partitioner_importable()
        self.cfg = config
        self._slots = threading.Semaphore(max(config.max_concurrent, 0))
        self._lock = threading.Lock()
        self._plans: "OrderedDict[str, object]" = OrderedDict()
        self._stages: "OrderedDict[tuple, object]" = OrderedDict()
        self._stage_factory = stage_factory
        # Test hook: runs right before the forward pass (e.g. a delay).
        self._before_compute = before_compute

    # ---- RPC -----------------------------------------------------------

    def ExecuteStage(self, request, context):  # noqa: N802 -- gRPC-generated method name
        t0 = time.perf_counter()
        if not self._slots.acquire(blocking=False):
            logger.info("stage %d request %s: BUSY (%d concurrent)", request.stage_index, request.request_id,
                        self.cfg.max_concurrent)
            return self._response(request, t0, error=_Stop(
                pb.STAGE_STATUS_BUSY, "BUSY", f"worker is at its concurrency limit ({self.cfg.max_concurrent}); not executed"))
        timing = {"validate": 0.0, "compute": 0.0}
        try:
            output = self._handle(request, context, timing)
            resp = self._response(request, t0, output=output, timing=timing)
        except _Stop as stop:
            resp = self._response(request, t0, error=stop, timing=timing)
        except Exception as exc:  # a bug in this service; reported, not hidden
            logger.exception("stage %d request %s: internal error", request.stage_index, request.request_id)
            resp = self._response(request, t0, timing=timing, error=_Stop(
                pb.STAGE_STATUS_FAILED, "INTERNAL", f"{type(exc).__name__}: {exc}", executed=True))
        finally:
            self._slots.release()
        logger.info("stage %d request %s: %s %s in %.4fs (compute %.4fs)%s", request.stage_index, request.request_id,
                    pb.StageStatus.Name(resp.status), self._shape_note(request, resp), resp.timing.total_seconds,
                    resp.timing.compute_seconds, f" [{resp.error.code}: {resp.error.message}]" if resp.error.code else "")
        return resp

    # ---- handling ------------------------------------------------------

    def _handle(self, request, context, timing):
        from partitioner.execution import wire
        from partitioner.execution.interface import InterfaceError, TOKEN_IDS

        tv = time.perf_counter()
        if request.protocol_version != PROTOCOL_VERSION:
            raise _reject("PROTOCOL_VERSION", f"protocol_version {int(request.protocol_version)} is not supported "
                                              f"(this worker speaks {int(PROTOCOL_VERSION)})")
        missing = [f for f in ("request_id", "plan_id", "plan_json", "model_id", "config_fingerprint", "target_worker_id")
                   if not getattr(request, f)]
        if not request.HasField("input"):
            missing.append("input")
        problems = [f"missing field {f}" for f in missing]
        if request.stage_index < 0 or not 0 <= request.unit_start < request.unit_end:
            problems.append(f"invalid stage_index {request.stage_index} / unit range "
                            f"[{request.unit_start}, {request.unit_end})")
        if problems:
            raise _reject("MALFORMED_REQUEST", "malformed request", problems)
        if request.target_worker_id != self.cfg.worker_id:
            raise _reject("WRONG_WORKER", f"request is for worker {request.target_worker_id!r}, this is "
                                          f"{self.cfg.worker_id!r}")
        remaining = context.time_remaining()
        if remaining is None or remaining > 1e9:
            raise _reject("DEADLINE", "the request has no deadline; a deadline is required")
        if remaining < self.cfg.min_deadline_seconds:
            raise _reject("DEADLINE", f"only {remaining:.3f}s left before the deadline "
                                      f"(< {self.cfg.min_deadline_seconds}s minimum)")

        loaded = self._plan(request)
        ss = self._check_assignment(request, loaded)

        # Tensor metadata vs. stage and limits -- before allocating.
        kind, dtype, shape = wire.describe(request.input)
        invalid = wire.check_metadata(request.input, self.cfg.max_tensor_bytes)
        limits = []
        want_dtype = "int64" if ss.input_kind == TOKEN_IDS else ss.activation_dtype
        if kind != ss.input_kind:
            invalid.append(f"input kind {kind!r} != {ss.input_kind!r} expected by stage {ss.stage_index}")
        if dtype != want_dtype:
            invalid.append(f"input dtype {dtype!r} != {want_dtype!r} expected by stage {ss.stage_index}")
        want_from = -1 if ss.stage_index == 0 else ss.stage_index - 1
        if request.input.produced_by_stage != want_from:
            invalid.append(f"input produced_by_stage {request.input.produced_by_stage} != {want_from}")
        spec, wl = loaded.spec, loaded.plan["workload"]
        want_rank = 2 if ss.input_kind == TOKEN_IDS else 3
        if len(shape) != want_rank:
            invalid.append(f"input rank {len(shape)} != {want_rank} for {ss.input_kind}")
        elif want_rank == 3 and shape[2] != spec.hidden_size:
            invalid.append(f"input hidden size {shape[2]} != {spec.hidden_size}")
        if len(shape) >= 2 and all(d > 0 for d in shape):
            batch, tokens = shape[0], shape[1]
            max_seq = min(spec.max_positions, self.cfg.max_sequence or spec.max_positions)
            if batch > self.cfg.max_batch:
                limits.append(f"batch {batch} > worker limit {self.cfg.max_batch}")
            if batch > wl["batch_size"]:
                limits.append(f"batch {batch} > the plan's workload batch_size {wl['batch_size']}")
            if tokens > max_seq:
                limits.append(f"{tokens} tokens > limit {max_seq}")
            if tokens > wl["prompt_length"]:
                limits.append(f"{tokens} tokens > the plan's workload prompt_length {wl['prompt_length']}")
            width = spec.vocab_size if ss.output_kind == "logits" else spec.hidden_size
            out_bytes = batch * tokens * width * 4
            if out_bytes > self.cfg.max_tensor_bytes:
                limits.append(f"output would be {out_bytes} bytes > limit {self.cfg.max_tensor_bytes}")
        if any("bytes > limit" in p for p in invalid):
            limits.extend(p for p in invalid if "bytes > limit" in p)
            invalid = [p for p in invalid if "bytes > limit" not in p]
        if invalid:
            raise _reject("INVALID_TENSOR", "input tensor rejected", invalid + limits)
        if limits:
            raise _reject("LIMIT_EXCEEDED", "request exceeds a resource limit", limits)
        try:
            activation = wire.decode(request.input, self.cfg.max_tensor_bytes, request.request_id)
        except wire.WireError as exc:
            raise _reject(exc.code, "input tensor rejected", exc.problems) from exc

        stage = self._stage(loaded, ss)
        timing["validate"] = time.perf_counter() - tv
        if self._before_compute is not None:
            self._before_compute(request, context)
        tc = time.perf_counter()
        try:
            out = stage.forward(activation)
        except InterfaceError as exc:  # the stage's own input checks run before any compute
            raise _reject("INVALID_TENSOR", str(exc)) from exc
        except Exception as exc:
            raise _Stop(pb.STAGE_STATUS_FAILED, "EXECUTION_ERROR", f"{type(exc).__name__}: {exc}", executed=True) from exc
        finally:
            timing["compute"] = time.perf_counter() - tc
        remaining = context.time_remaining()
        if not context.is_active() or (remaining is not None and remaining <= 0):
            raise _Stop(pb.STAGE_STATUS_DEADLINE_EXCEEDED, "DEADLINE",
                        "the stage finished after the request's deadline; output discarded", executed=True)
        return wire.encode(out)

    def _plan(self, request):
        from partitioner import plan as plan_mod
        from partitioner.execution.loader import REMOTE_MODE, ExecutionRejected, load_for_execution

        try:
            plan = json.loads(request.plan_json.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise _reject("PLAN_INVALID", f"plan_json is not valid UTF-8 JSON: {exc}") from exc
        if not isinstance(plan, dict):
            raise _reject("PLAN_INVALID", "plan_json must be a JSON object")
        if plan.get("inputs_fingerprint") != request.plan_id or plan_mod.fingerprint(plan) != request.plan_id:
            raise _reject("PLAN_INVALID", "plan_json does not hash to plan_id (or its inputs_fingerprint differs)")
        with self._lock:
            cached = self._plans.get(request.plan_id)
            if cached is not None:
                self._plans.move_to_end(request.plan_id)
                return cached
        try:
            loaded = load_for_execution(plan, mode=REMOTE_MODE, workers=None, local_hostname=self.cfg.hostname)
        except ExecutionRejected as exc:
            raise _reject("PLAN_INVALID", "plan rejected", exc.problems) from exc
        with self._lock:
            self._plans[request.plan_id] = loaded
            while len(self._plans) > self.cfg.plan_cache_size:
                self._plans.popitem(last=False)
        return loaded

    def _check_assignment(self, request, loaded):
        stages = loaded.plan["stages"]
        idx = request.stage_index
        if not 0 <= idx < len(stages):
            raise _reject("UNASSIGNED_STAGE", f"plan has no stage {idx} (it has {len(stages)})")
        st = stages[idx]
        if st["worker_id"] != self.cfg.worker_id:
            raise _reject("UNASSIGNED_STAGE", f"stage {idx} is assigned to worker {st['worker_id']!r}, not to this "
                                              f"worker {self.cfg.worker_id!r}")
        problems = []
        if st.get("hostname") != self.cfg.hostname:
            problems.append(f"stage {idx} was planned for host {st.get('hostname')!r}, this is {self.cfg.hostname!r}")
        if st.get("profile_source") != "REAL":
            problems.append(f"stage {idx} worker profile is {st.get('profile_source')!r}, not REAL")
        if list(st["unit_range"]) != [request.unit_start, request.unit_end]:
            problems.append(f"requested units [{request.unit_start}, {request.unit_end}) != the plan's {st['unit_range']}")
        if request.model_id != loaded.plan["model"]["model_id"]:
            problems.append(f"model_id {request.model_id!r} != the plan's {loaded.plan['model']['model_id']!r}")
        if request.config_fingerprint != loaded.plan["model"]["config_fingerprint"]:
            problems.append("config_fingerprint does not match the plan's model")
        if problems:
            raise _reject("STAGE_MISMATCH", "request does not match this worker's assignment in the plan", problems)
        return loaded.stage_specs[idx]

    def _stage(self, loaded, ss):
        key = (loaded.plan_info["plan_id"], ss.stage_index)
        with self._lock:
            stage = self._stages.get(key)
            if stage is not None:
                self._stages.move_to_end(key)
                return stage
        factory = self._stage_factory or loaded.model.build_stage
        stage = factory(ss, loaded.spec)
        with self._lock:
            self._stages[key] = stage
            while len(self._stages) > self.cfg.stage_cache_size:
                self._stages.popitem(last=False)
        return stage

    # ---- responses -----------------------------------------------------

    def _response(self, request, t0, output=None, error: Optional[_Stop] = None, timing=None):
        timing = timing or {}
        kw = dict(
            protocol_version=PROTOCOL_VERSION,
            request_id=request.request_id,
            worker_id=self.cfg.worker_id,
            stage_index=request.stage_index,
            status=pb.STAGE_STATUS_OK if error is None else error.status,
            timing=pb.StageTiming(validate_seconds=timing.get("validate", 0.0), compute_seconds=timing.get("compute", 0.0),
                                  total_seconds=time.perf_counter() - t0),
            worker_hostname=self.cfg.hostname,
            worker_pid=os.getpid(),
        )
        if output is not None:
            kw["output"] = output
        if error is not None:
            kw["error"] = pb.StageError(code=error.code, message=str(error), problems=error.problems, executed=error.executed)
        return pb.ExecuteStageResponse(**kw)

    @staticmethod
    def _shape_note(request, resp) -> str:
        i = list(request.input.shape) if request.HasField("input") else None
        o = list(resp.output.shape) if resp.HasField("output") else None
        return f"(input {i} -> output {o})"
