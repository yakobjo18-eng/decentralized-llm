"""Phase 6 coordinator: execute_plan_remote() and `execute --remote`.

Part A uses StageExecution servers running inside THIS test process
(real gRPC over 127.0.0.1, fast) to cover retry, failure and rejection
paths. Part B starts SEPARATE worker processes (`python -m
worker.execution`) and runs the pipeline across them over real gRPC.
Everything here runs on one computer: none of it is a physical
multi-machine test.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time

import pytest
import torch

from conftest import make_worker, tiny_plan
from partitioner.cli import main
from partitioner.execution import tiny_gpt2 as t
from partitioner.execution.remote import REMOTE_REFERENCE_ATOL, RemoteOptions, execute_plan_remote, parse_endpoint
from remote_helpers import IDS, InProcessWorker, WorkerProcess, free_port, plan_for

FAST = RemoteOptions(stage_timeout_seconds=30, connect_timeout_seconds=0.5, max_attempts=3, retry_backoff_seconds=0.05)


@pytest.fixture
def workers():
    started = {}

    def start(*ids, **cfg):
        for wid in ids:
            started[wid] = InProcessWorker(wid, **cfg)
        return {wid: started[wid].endpoint for wid in ids}

    yield start, started
    for w in started.values():
        w.stop()


# =========================== Part A: in-process servers ===========================


def test_two_stage_pipeline_matches_the_local_reference(workers):
    start, _ = workers
    eps = start("w-a", "w-b")
    plan = plan_for(["w-a", "w-b"])
    r = execute_plan_remote(plan, IDS, eps, options=FAST, include_logits=True)
    assert r["status"] == "succeeded", r["error"]
    assert r["execution_mode"] == "remote" and r["protocol_version"] == 1
    assert [s["worker_id"] for s in r["stages"]] == ["w-a", "w-b"]
    assert r["stages"][0]["output"] == {"kind": "hidden_states", "shape": [2, 8, 32], "dtype": "fp32"}
    assert r["output"]["shape"] == [2, 8, 128] and r["output"]["next_token_ids"] == [60, 96]
    ref = t.reference_forward(torch.tensor(IDS))
    assert (torch.tensor(r["output"]["logits"]) - ref).abs().max().item() <= REMOTE_REFERENCE_ATOL
    assert r["reference_check"]["passed"] and r["reference_check"]["max_abs_diff"] <= REMOTE_REFERENCE_ATOL
    assert all(s["attempts"] == [s["attempts"][0]] and s["attempts"][0]["outcome"] == "STAGE_STATUS_OK" for s in r["stages"])
    # Both servers live in this test process: honestly NOT distributed.
    assert r["distinct_worker_processes"] == 1 and r["distributed"] is False
    assert r["plan"]["plan_id"] == plan["inputs_fingerprint"]


def test_three_stage_pipeline(workers):
    start, _ = workers
    eps = start("w-a", "w-b", "w-c")
    r = execute_plan_remote(plan_for(["w-a", "w-b", "w-c"]), IDS, eps, options=FAST)
    assert r["status"] == "succeeded", r["error"]
    assert len(r["stages"]) == 3 and r["reference_check"]["passed"]


def test_missing_endpoint_is_rejected_before_dispatch(workers):
    start, _ = workers
    eps = start("w-a")
    r = execute_plan_remote(plan_for(["w-a", "w-b"]), IDS, eps, options=FAST)
    assert r["status"] == "rejected" and r["error"]["phase"] == "endpoints" and r["stages"] == []
    assert any("no endpoint given for worker 'w-b'" in p for p in r["error"]["problems"])


def test_simulated_plan_is_rejected(workers):
    start, _ = workers
    eps = start("w-a", "w-b")
    sim = tiny_plan(t.spec(), [make_worker("w-a"), make_worker("w-b")], allow_unused_workers=False)
    r = execute_plan_remote(sim, IDS, eps, options=FAST)
    assert r["status"] == "rejected"
    assert any("SIMULATED workers: refusing remote execution" in p for p in r["error"]["problems"])


def test_input_over_limits_is_rejected_before_dispatch(workers):
    start, started = workers
    eps = start("w-a", "w-b")
    r = execute_plan_remote(plan_for(["w-a", "w-b"]), [[1] * 40], eps, options=FAST)
    assert r["status"] == "rejected" and r["error"]["phase"] == "resource_limits" and r["stages"] == []


def test_misrouted_stage_is_refused_by_the_worker(workers):
    """Endpoints swapped: w-a's stage reaches w-b's server, which refuses it."""
    start, _ = workers
    eps = start("w-a", "w-b")
    r = execute_plan_remote(plan_for(["w-a", "w-b"]), IDS, {"w-a": eps["w-b"], "w-b": eps["w-a"]}, options=FAST)
    assert r["status"] == "failed" and r["error"]["code"] == "WRONG_WORKER"
    assert r["error"]["executed"] is False and len(r["error"]["attempts"]) == 1  # a rejection is not retried
    assert r["stages"] == [] and r["error"]["stage_index"] == 0


def test_unreachable_worker_is_retried_then_reported(workers):
    start, _ = workers
    eps = start("w-a")
    eps["w-b"] = f"127.0.0.1:{free_port()}"  # nothing listens here
    r = execute_plan_remote(plan_for(["w-a", "w-b"]), IDS, eps, options=FAST)
    assert r["status"] == "failed" and r["error"]["code"] == "RETRIES_EXHAUSTED" and r["error"]["stage_index"] == 1
    attempts = r["error"]["attempts"]
    assert [a["outcome"] for a in attempts] == ["unreachable"] * 3 and all(a["executed"] is False for a in attempts)
    assert [s["stage_index"] for s in r["stages"]] == [0]  # stage 0 completed on w-a
    assert r["failed_stage"]["worker_id"] == "w-b"


def test_busy_worker_is_retried_until_it_has_capacity(workers):
    start, started = workers
    eps = start("w-a")
    eps.update(start("w-b", max_concurrent=1))
    slots = started["w-b"].servicer._slots
    assert slots.acquire(blocking=False)  # occupy w-b's only slot...
    threading.Timer(0.4, slots.release).start()  # ...for 0.4 s
    opts = RemoteOptions(connect_timeout_seconds=0.5, max_attempts=6, retry_backoff_seconds=0.1)
    r = execute_plan_remote(plan_for(["w-a", "w-b"]), IDS, eps, options=opts)
    assert r["status"] == "succeeded", r["error"]
    outcomes = [a["outcome"] for a in r["stages"][1]["attempts"]]
    assert outcomes[0] == "STAGE_STATUS_BUSY" and outcomes[-1] == "STAGE_STATUS_OK" and len(outcomes) <= 6


def test_permanently_busy_worker_exhausts_bounded_retries(workers):
    start, _ = workers
    eps = start("w-a")
    eps.update(start("w-b", max_concurrent=0))
    r = execute_plan_remote(plan_for(["w-a", "w-b"]), IDS, eps, options=FAST)
    assert r["status"] == "failed" and r["error"]["code"] == "RETRIES_EXHAUSTED" and r["error"]["executed"] is False
    assert [a["outcome"] for a in r["error"]["attempts"]] == ["STAGE_STATUS_BUSY"] * 3


def test_deadline_exceeded_is_ambiguous_and_not_retried(workers):
    start, _ = workers
    eps = start("w-a")
    eps.update(start("w-b", before_compute=lambda r, c: time.sleep(1.5)))
    opts = RemoteOptions(stage_timeout_seconds=0.6, connect_timeout_seconds=0.5, max_attempts=3, retry_backoff_seconds=0)
    r = execute_plan_remote(plan_for(["w-a", "w-b"]), IDS, eps, options=opts)
    assert r["status"] == "failed" and r["error"]["code"] == "AMBIGUOUS_FAILURE"
    assert "DEADLINE_EXCEEDED" in r["error"]["message"] and "not retried" in r["error"]["message"]
    assert r["error"]["executed"] is None and len(r["error"]["attempts"]) == 1
    time.sleep(1.2)  # let the slow handler finish before the server stops


def test_stage_failure_is_not_retried(workers):
    class Broken:
        def __init__(self, ss, spec):
            pass

        def forward(self, activation):
            raise RuntimeError("stage crashed on the worker")

    start, _ = workers
    eps = start("w-a")
    eps.update(start("w-b", stage_factory=Broken))
    r = execute_plan_remote(plan_for(["w-a", "w-b"]), IDS, eps, options=FAST)
    assert r["status"] == "failed" and r["error"]["code"] == "EXECUTION_ERROR" and r["error"]["executed"] is True
    assert "stage crashed on the worker" in r["error"]["message"] and len(r["error"]["attempts"]) == 1


def test_wrong_remote_numbers_fail_the_reference_check(workers):
    class Off(t.TinyGPT2Stage):
        def forward(self, activation):
            out = super().forward(activation)
            return type(out)(out.meta, out.tensor + 0.01)

    start, _ = workers
    eps = start("w-a")
    eps.update(start("w-b", stage_factory=lambda ss, spec: Off(ss, spec)))
    r = execute_plan_remote(plan_for(["w-a", "w-b"]), IDS, eps, options=FAST)
    assert r["status"] == "failed" and r["error"]["phase"] == "reference_check"
    assert r["reference_check"]["passed"] is False and r["reference_check"]["max_abs_diff"] > REMOTE_REFERENCE_ATOL


@pytest.mark.parametrize("text", ["w-a", "w-a=host", "w-a=:50200", "=h:1", "w-a=h:0", "w-a=h:x"])
def test_endpoint_parsing(text):
    with pytest.raises(ValueError):
        parse_endpoint(text)
    assert parse_endpoint("w-a=192.168.1.20:50200") == ("w-a", "192.168.1.20:50200")


def test_invalid_remote_options_are_rejected():
    r = execute_plan_remote(plan_for(["w-a", "w-b"]), IDS, {}, options=RemoteOptions(max_attempts=0))
    assert r["status"] == "rejected" and r["error"]["phase"] == "remote_options"


# ---- CLI --------------------------------------------------------------------


def test_cli_remote_success_and_failures(workers, tmp_path, capsys):
    start, _ = workers
    eps = start("w-a", "w-b")
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan_for(["w-a", "w-b"])))
    args = ["execute", str(path), "--tokens", "1,2,3,4,5,6,7,8;9,10,11,12,13,14,15,16", "--remote",
            "--endpoint", f"w-a={eps['w-a']}", "--endpoint", f"w-b={eps['w-b']}", "--connect-timeout", "0.5"]
    assert main(args) == 0
    cap = capsys.readouterr()
    result = json.loads(cap.out)
    assert result["status"] == "succeeded" and result["execution_mode"] == "remote"
    assert "SUCCEEDED (remote, distributed: false)" in cap.err  # in-process servers: one process

    assert main(args[:-4] + ["--connect-timeout", "0.5"]) == 1  # no endpoint for w-b
    assert json.loads(capsys.readouterr().out)["error"]["phase"] == "endpoints"


@pytest.mark.parametrize("extra, message", [
    (["--remote", "--endpoint", "w-a"], "WORKER_ID=HOST:PORT"),
    (["--emulate-stages-locally", "--endpoint", "w-a=h:1"], "only used with --remote"),
    (["--remote", "--emulate-stages-locally"], "choose exactly one"),
    (["--remote", "--endpoint", "w-a=h:1", "--endpoint", "w-a=h:2"], "given twice"),
])
def test_cli_remote_usage_errors(tmp_path, capsys, extra, message):
    assert main(["execute", str(tmp_path / "p.json"), "--tokens", "1", *extra]) == 2
    assert message in capsys.readouterr().err


# ====================== Part B: separate worker processes ======================


@pytest.fixture(scope="module")
def processes(tmp_path_factory):
    """Three real `python -m worker.execution` processes on 127.0.0.1,
    started in parallel (each imports torch + transformers)."""
    tmp = tmp_path_factory.mktemp("workers")
    procs = {wid: WorkerProcess(tmp, wid) for wid in ("p-a", "p-b", "p-c")}
    try:
        for p in procs.values():
            p.wait_ready()
        yield procs
    finally:
        for p in procs.values():
            p.kill()


def test_two_worker_processes_over_grpc(processes):
    eps = {w: processes[w].endpoint for w in ("p-a", "p-b")}
    plan = plan_for(["p-a", "p-b"])
    r = execute_plan_remote(plan, IDS, eps, options=RemoteOptions(connect_timeout_seconds=5), include_logits=True)
    assert r["status"] == "succeeded", r["error"]
    pids = [s["worker_reported"]["pid"] for s in r["stages"]]
    assert pids == [processes["p-a"].proc.pid, processes["p-b"].proc.pid] and os.getpid() not in pids
    assert r["distinct_worker_processes"] == 2 and r["distributed"] is True
    assert r["distinct_worker_hostnames"] == [socket.gethostname()]
    assert "NOT a multi-machine run" in r["distributed_note"]
    ref = t.reference_forward(torch.tensor(IDS))
    diff = (torch.tensor(r["output"]["logits"]) - ref).abs().max().item()
    assert diff <= REMOTE_REFERENCE_ATOL and r["reference_check"]["max_abs_diff"] == diff
    assert r["output"]["next_token_ids"] == [60, 96]


def test_three_worker_processes_over_grpc(processes):
    eps = {w: p.endpoint for w, p in processes.items()}
    r = execute_plan_remote(plan_for(["p-a", "p-b", "p-c"]), IDS, eps, options=RemoteOptions(connect_timeout_seconds=5))
    assert r["status"] == "succeeded", r["error"]
    assert r["distinct_worker_processes"] == 3 and r["reference_check"]["passed"]
    assert [h["kind"] for h in (s["output"] for s in r["stages"])] == ["hidden_states", "hidden_states", "logits"]


def test_cli_against_worker_processes(processes, tmp_path, capsys):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan_for(["p-a", "p-b"])))
    rc = main(["execute", str(path), "--tokens", "5,6,7", "--remote", "--endpoint", f"p-a={processes['p-a'].endpoint}",
               "--endpoint", f"p-b={processes['p-b'].endpoint}"])
    cap = capsys.readouterr()
    assert rc == 0, cap.out
    assert "SUCCEEDED (remote, distributed: true)" in cap.err


def test_killed_worker_process_fails_its_stage_cleanly(processes):
    """Runs last in this module: kills p-c for good."""
    plan = plan_for(["p-a", "p-c"])
    eps = {"p-a": processes["p-a"].endpoint, "p-c": processes["p-c"].endpoint}
    processes["p-c"].kill()
    r = execute_plan_remote(plan, IDS, eps, options=FAST)
    assert r["status"] == "failed" and r["error"]["stage_index"] == 1 and r["error"]["code"] == "RETRIES_EXHAUSTED"
    assert all(a["outcome"] == "unreachable" and a["executed"] is False for a in r["error"]["attempts"])
    assert len(r["error"]["attempts"]) == FAST.max_attempts
    assert [s["stage_index"] for s in r["stages"]] == [0]
