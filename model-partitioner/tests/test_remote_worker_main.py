"""Full worker processes (`python -m worker.main`) with
execution_enabled, registered with the REAL Go orchestrator: they
register, keep heartbeating, run a two-stage pipeline for the
coordinator, and shut down cleanly. Shows the StageExecution service
leaves registration and heartbeats alone.

Local processes on one computer, not a multi-machine test. Skipped if
orchestrator.exe has not been built.
"""

from __future__ import annotations

import json
import signal
import subprocess
import time
import urllib.request

import pytest

from conftest import REPO
from partitioner.execution.remote import RemoteOptions, execute_plan_remote
from remote_helpers import IDS, WorkerProcess, free_port, plan_for

ORCH_EXE = REPO / "orchestrator" / "orchestrator.exe"
pytestmark = pytest.mark.skipif(not ORCH_EXE.exists(), reason="orchestrator.exe not built")
READY = "StageExecution (protocol v1, UNAUTHENTICATED) listening on"


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as resp:
        return json.loads(resp.read().decode())


def _wait(pred, timeout):
    end = time.time() + timeout
    while time.time() < end:
        try:
            if pred():
                return True
        except Exception:
            pass
        time.sleep(0.3)
    return False


def test_worker_main_with_execution_registers_heartbeats_and_executes(tmp_path):
    rest, grpc_port = free_port(), free_port()
    (tmp_path / "orch.yaml").write_text(f'rest_address: "127.0.0.1:{rest}"\ngrpc_address: "127.0.0.1:{grpc_port}"\n'
                                        'log_level: "info"\nheartbeat_timeout_seconds: 3\n')
    orch = subprocess.Popen([str(ORCH_EXE), "--config", str(tmp_path / "orch.yaml")],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    workers = {}
    try:
        assert _wait(lambda: _get(f"http://127.0.0.1:{rest}/health")["status"] == "ok", 10)
        health_ports = {}
        for wid in ("m-a", "m-b"):
            health_ports[wid] = free_port()
            extra = (f'orchestrator_address: "127.0.0.1:{grpc_port}"\nlisten_host: "127.0.0.1"\n'
                     f'listen_port: {health_ports[wid]}\nheartbeat_interval_seconds: 0.5\nbackend_compute_test: false\n'
                     'execution_enabled: true\nexecution_listen_port: 0\n')
            workers[wid] = WorkerProcess(tmp_path, wid, entry="worker.main", extra_yaml=extra, ready_marker=READY)
        eps = {wid: w.wait_ready() for wid, w in workers.items()}

        def registered():
            by_id = {w["id"]: w for w in _get(f"http://127.0.0.1:{rest}/api/workers")}
            return all(by_id.get(w, {}).get("online") for w in workers)

        assert _wait(registered, 30), "workers did not register"
        by_id = {w["id"]: w for w in _get(f"http://127.0.0.1:{rest}/api/workers")}
        for wid in workers:
            # Registration still advertises the Health server, not the execution port.
            assert by_id[wid]["address"] == f"127.0.0.1:{health_ports[wid]}"
            assert by_id[wid]["address"] != eps[wid]

        r = execute_plan_remote(plan_for(["m-a", "m-b"]), IDS, eps, options=RemoteOptions(connect_timeout_seconds=5))
        assert r["status"] == "succeeded", r["error"]
        assert r["distinct_worker_processes"] == 2 and r["reference_check"]["passed"]

        time.sleep(2)  # heartbeats keep flowing during and after execution
        by_id = {w["id"]: w for w in _get(f"http://127.0.0.1:{rest}/api/workers")}
        assert all(by_id[w]["online"] and by_id[w]["last_heartbeat_age_seconds"] < 2.5 for w in workers)

        # Graceful stop: deregisters and stops both servers.
        workers["m-a"].proc.send_signal(signal.CTRL_BREAK_EVENT)
        assert workers["m-a"].proc.wait(timeout=20) == 0
        assert any("worker shutdown complete" in line for line in workers["m-a"].log)
        assert _wait(lambda: "m-a" not in {w["id"] for w in _get(f"http://127.0.0.1:{rest}/api/workers")}, 10)
    finally:
        for w in workers.values():
            w.kill()
        orch.kill()
        orch.wait(timeout=10)
