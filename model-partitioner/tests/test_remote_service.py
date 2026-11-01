"""Worker-side StageExecution service, called over real gRPC (server in
this process): protocol version, malformed requests, worker identity,
plan and assignment checks, tensor validation, limits, deadlines,
concurrency, execution failures, and what gets logged."""

from __future__ import annotations

import copy
import json
import logging
import socket
import time

import grpc
import pytest

from remote_helpers import IDS, InProcessWorker, hidden_payload, plan_for, token_payload
from conftest import make_worker, refingerprint, tiny_plan
from partitioner.execution import tiny_gpt2 as t
from partitioner.execution import wire

pb = wire.pb2()


@pytest.fixture(scope="module")
def plan():
    return plan_for(["w-a", "w-b"])  # w-a: [0, 3), w-b: [3, 7)


@pytest.fixture
def worker(request):
    servers = []

    def start(worker_id="w-a", **cfg):
        w = InProcessWorker(worker_id, **cfg)
        servers.append(w)
        return w, w.stub()

    yield start
    for w in servers:
        w.stop()


def req(plan, stage=0, target="w-a", payload=None, **kw):
    st = plan["stages"][stage]
    base = dict(protocol_version=pb.PROTOCOL_VERSION_1, request_id="req-1", plan_id=plan["inputs_fingerprint"],
                plan_json=json.dumps(plan).encode(), model_id=plan["model"]["model_id"],
                config_fingerprint=plan["model"]["config_fingerprint"], target_worker_id=target, stage_index=stage,
                unit_start=st["unit_range"][0], unit_end=st["unit_range"][1],
                input=payload if payload is not None else (token_payload() if stage == 0 else hidden_payload()))
    base.update(kw)
    return pb.ExecuteStageRequest(**base)


def call(stub, request, timeout=30):
    return stub.ExecuteStage(request, timeout=timeout)


def assert_rejected(resp, code, fragment=""):
    assert resp.status == pb.STAGE_STATUS_REJECTED, pb.StageStatus.Name(resp.status)
    assert resp.error.code == code, (resp.error.code, resp.error.message, list(resp.error.problems))
    assert resp.error.executed is False and not resp.HasField("output")
    text = resp.error.message + " " + " ".join(resp.error.problems)
    assert fragment in text, text


# ---- success ---------------------------------------------------------------


def test_assigned_stage_executes(plan, worker):
    _, stub = worker()
    resp = call(stub, req(plan))
    assert resp.status == pb.STAGE_STATUS_OK, resp.error
    assert resp.worker_id == "w-a" and resp.stage_index == 0 and resp.request_id == "req-1"
    assert resp.worker_hostname == socket.gethostname() and resp.worker_pid > 0
    assert resp.protocol_version == pb.PROTOCOL_VERSION_1
    out = wire.decode(resp.output, 1 << 20, "req-1")
    assert out.meta.kind == "hidden_states" and out.meta.shape == (2, 8, 32) and out.meta.produced_by_stage == 0
    assert resp.timing.total_seconds >= resp.timing.compute_seconds > 0
    # Same numbers as running units [0, 3) locally.
    local = t.TinyGPT2Stage(t.make_stage_spec(t.spec(), 0, 2, 0, 3)).forward(wire.decode(token_payload(), 1 << 20, "req-1"))
    assert (out.tensor - local.tensor).abs().max().item() == 0.0


def test_last_stage_returns_logits(plan, worker):
    _, stub = worker("w-b")
    resp = call(stub, req(plan, stage=1, target="w-b"))
    assert resp.status == pb.STAGE_STATUS_OK, resp.error
    assert wire.decode(resp.output, 1 << 20).meta.shape == (2, 8, 128)


# ---- protocol and request shape -----------------------------------------------


@pytest.mark.parametrize("version", [pb.PROTOCOL_VERSION_UNSPECIFIED, 2, 7])
def test_unsupported_protocol_version(plan, worker, version):
    _, stub = worker()
    resp = call(stub, req(plan, protocol_version=version))
    assert_rejected(resp, "PROTOCOL_VERSION", f"protocol_version {version} is not supported")
    assert resp.protocol_version == pb.PROTOCOL_VERSION_1  # the worker states what it speaks


@pytest.mark.parametrize("change, fragment", [
    (dict(plan_json=b""), "missing field plan_json"),
    (dict(request_id=""), "missing field request_id"),
    (dict(target_worker_id=""), "missing field target_worker_id"),
    (dict(unit_start=3, unit_end=3), "unit range"),
    (dict(stage_index=-1), "invalid stage_index"),
])
def test_malformed_requests(plan, worker, change, fragment):
    _, stub = worker()
    assert_rejected(call(stub, req(plan, **change)), "MALFORMED_REQUEST", fragment)


def test_missing_input_tensor(plan, worker):
    _, stub = worker()
    r = req(plan)
    r.ClearField("input")
    assert_rejected(call(stub, r), "MALFORMED_REQUEST", "missing field input")


# ---- identity, plan and assignment -------------------------------------------


def test_request_for_another_worker_is_refused(plan, worker):
    _, stub = worker("w-a")
    assert_rejected(call(stub, req(plan, target="w-b")), "WRONG_WORKER", "is for worker 'w-b'")


def test_unassigned_stage_is_refused(plan, worker):
    _, stub = worker("w-a")
    resp = call(stub, req(plan, stage=1, target="w-a"))  # stage 1 belongs to w-b
    assert_rejected(resp, "UNASSIGNED_STAGE", "assigned to worker 'w-b', not to this worker 'w-a'")


def test_nonexistent_stage_is_refused(plan, worker):
    _, stub = worker()
    resp = call(stub, req(plan, stage=0, stage_index=5))
    assert_rejected(resp, "UNASSIGNED_STAGE", "plan has no stage 5")


def test_plan_must_hash_to_plan_id(plan, worker):
    _, stub = worker()
    edited = copy.deepcopy(plan)
    edited["workload"]["prompt_length"] = 4  # content changed, fingerprint not
    assert_rejected(call(stub, req(plan, plan_json=json.dumps(edited).encode())), "PLAN_INVALID", "does not hash")
    assert_rejected(call(stub, req(plan, plan_id="0" * 64)), "PLAN_INVALID", "does not hash")
    assert_rejected(call(stub, req(plan, plan_json=b"\xff{")), "PLAN_INVALID", "not valid UTF-8 JSON")


def test_invalid_plan_is_refused_even_if_correctly_fingerprinted(plan, worker):
    _, stub = worker()
    bad = refingerprint(copy.deepcopy(plan))
    bad["stages"][0]["unit_range"][1] = 2  # gap at unit 2
    bad = refingerprint(bad)
    resp = call(stub, req(bad, plan_id=bad["inputs_fingerprint"], unit_end=2))
    assert_rejected(resp, "PLAN_INVALID", "gap or overlap")


def test_simulated_plan_is_refused(worker):
    sim = tiny_plan(t.spec(), [make_worker("w-a"), make_worker("w-b")], allow_unused_workers=False)
    _, stub = worker()
    assert_rejected(call(stub, req(sim)), "PLAN_INVALID", "SIMULATED")


@pytest.mark.parametrize("change, fragment", [
    (dict(unit_end=2), "requested units [0, 2)"),
    (dict(model_id="gpt2"), "model_id 'gpt2'"),
    (dict(config_fingerprint="f" * 64), "config_fingerprint"),
])
def test_request_must_match_the_assignment(plan, worker, change, fragment):
    _, stub = worker()
    assert_rejected(call(stub, req(plan, **change)), "STAGE_MISMATCH", fragment)


def test_worker_on_another_host_is_refused(plan, worker):
    _, stub = worker("w-a", hostname="some-other-host")
    assert_rejected(call(stub, req(plan)), "STAGE_MISMATCH", "planned for host")


# ---- tensors and limits ---------------------------------------------------------


@pytest.mark.parametrize("make, fragment", [
    (lambda: token_payload(), "input kind 'token_ids' != 'hidden_states'"),
    (lambda: hidden_payload(hidden=16), "hidden size 16 != 32"),
    (lambda: hidden_payload(producer=-1), "produced_by_stage -1 != 0"),
])
def test_stage_incompatible_tensors(plan, worker, make, fragment):
    _, stub = worker("w-b")
    assert_rejected(call(stub, req(plan, stage=1, target="w-b", payload=make())), "INVALID_TENSOR", fragment)


def test_corrupt_payload(plan, worker):
    _, stub = worker("w-b")
    p = hidden_payload()
    p.data = p.data[:-4]
    assert_rejected(call(stub, req(plan, stage=1, target="w-b", payload=p)), "INVALID_TENSOR", "payload has")
    p = hidden_payload()
    p.dtype = pb.TENSOR_DTYPE_INT64
    p.data = p.data + p.data  # right length for int64, wrong dtype for the stage
    assert_rejected(call(stub, req(plan, stage=1, target="w-b", payload=p)), "INVALID_TENSOR", "dtype 'int64'")


def test_out_of_vocabulary_tokens_are_rejected_not_executed(plan, worker):
    _, stub = worker()
    assert_rejected(call(stub, req(plan, payload=token_payload([[1, 200]]))), "INVALID_TENSOR", "token ids must be in")


def test_batch_over_the_worker_limit(plan, worker):
    _, stub = worker(max_batch=1)
    assert_rejected(call(stub, req(plan)), "LIMIT_EXCEEDED", "batch 2 > worker limit 1")


def test_sequence_over_the_worker_limit(plan, worker):
    _, stub = worker(max_sequence=4)
    assert_rejected(call(stub, req(plan)), "LIMIT_EXCEEDED", "8 tokens > limit 4")


def test_sequence_beyond_the_planned_workload(worker):
    from partitioner.memory import Workload

    short = plan_for(["w-a", "w-b"], workload=Workload(batch_size=2, context_length=16, prompt_length=4))
    _, stub = worker()
    assert_rejected(call(stub, req(short)), "LIMIT_EXCEEDED", "the plan's workload prompt_length 4")


def test_output_over_the_tensor_limit(plan, worker):
    # Input hidden states 2x8x32x4 = 2048 B fit; logits 2x8x128x4 = 8192 B don't.
    _, stub = worker("w-b", max_tensor_bytes=4096)
    assert_rejected(call(stub, req(plan, stage=1, target="w-b")), "LIMIT_EXCEEDED", "output would be 8192 bytes > limit 4096")


def test_input_over_the_tensor_limit(plan, worker):
    _, stub = worker("w-b", max_tensor_bytes=1024)
    assert_rejected(call(stub, req(plan, stage=1, target="w-b")), "LIMIT_EXCEEDED", "> limit 1024")


# ---- deadlines, concurrency, failures ------------------------------------------


def test_deadline_is_required(plan, worker):
    _, stub = worker()
    assert_rejected(stub.ExecuteStage(req(plan)), "DEADLINE", "no deadline")


def test_deadline_too_short_is_refused(plan, worker):
    _, stub = worker(min_deadline_seconds=5.0)
    assert_rejected(call(stub, req(plan), timeout=2), "DEADLINE", "minimum")


def test_slow_stage_hits_the_deadline(plan, worker, caplog):
    _, stub = worker(before_compute=lambda r, c: time.sleep(1.0))
    call(stub, req(plan))  # warm the plan/stage caches so only the sleep is slow
    with caplog.at_level(logging.INFO, logger="worker.execution"):
        with pytest.raises(grpc.RpcError) as exc:
            call(stub, req(plan, request_id="late"), timeout=0.5)
        assert exc.value.code() == grpc.StatusCode.DEADLINE_EXCEEDED
        time.sleep(1.0)  # let the handler finish
    assert any("late" in r.getMessage() and "STAGE_STATUS_DEADLINE_EXCEEDED" in r.getMessage() for r in caplog.records)


def test_busy_worker_refuses_without_executing(plan, worker):
    _, stub = worker(max_concurrent=0)
    resp = call(stub, req(plan))
    assert resp.status == pb.STAGE_STATUS_BUSY and resp.error.code == "BUSY" and resp.error.executed is False


def test_execution_failure_is_structured(plan, worker):
    class Broken:
        def __init__(self, ss, spec):
            pass

        def forward(self, activation):
            raise RuntimeError("simulated stage crash")

    _, stub = worker(stage_factory=Broken)
    resp = call(stub, req(plan))
    assert resp.status == pb.STAGE_STATUS_FAILED and resp.error.code == "EXECUTION_ERROR"
    assert resp.error.executed is True and "simulated stage crash" in resp.error.message


def test_oversized_message_is_refused_by_grpc(plan, worker):
    _, stub = worker(max_message_bytes=4096)
    with pytest.raises(grpc.RpcError) as exc:
        call(stub, req(plan))
    assert exc.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED


def test_logs_carry_no_tensor_or_plan_contents(plan, worker, caplog):
    _, stub = worker()
    with caplog.at_level(logging.DEBUG):
        call(stub, req(plan))
        call(stub, req(plan, target="w-b"))
    text = "\n".join(r.getMessage() for r in caplog.records if r.name.startswith("worker"))
    assert "req-1" in text and "[2, 8]" in text
    assert "inputs_fingerprint" not in text and "\\x" not in text and "tensor(" not in text
    assert max(len(r.getMessage()) for r in caplog.records if r.name.startswith("worker")) < 600
