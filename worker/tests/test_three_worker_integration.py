"""Phase 2 multi-worker integration test, covering objective 5 in full:
starts the REAL orchestrator binary and three REAL worker processes
(each a separate OS process, its own config, its own worker_id and
port), verifies all three register, force-kills ONE of them to
simulate an unexpected disconnection (deliberately NOT graceful --
a real Deregister call would remove it immediately and never exercise
the staleness/timeout path this test is actually checking), waits for
it to be marked offline by the configured heartbeat timeout while
confirming the other two stay online throughout, then restarts the
killed worker and confirms it can register again.

This is local multi-process testing on Computer 1, not physical
multi-machine testing -- those are reported separately, per the
project's explicit instruction not to conflate the two. See
docs/DEVELOPMENT_STATUS.md.

Skipped if orchestrator.exe has not been built.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ORCH_EXE = REPO_ROOT / "orchestrator" / "orchestrator.exe"
WORKER_DIR = REPO_ROOT / "worker"

pytestmark = pytest.mark.skipif(
    not ORCH_EXE.exists(),
    reason="orchestrator.exe not built; run `go build -o orchestrator.exe ./cmd/orchestrator` in orchestrator/ first",
)


def _rest_get(url: str, timeout: float = 5.0):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _wait_for(predicate, timeout: float, interval: float = 0.3) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if predicate():
                return True
        except Exception:
            pass
        time.sleep(interval)
    return False


def _workers_by_id(rest_port: int) -> dict:
    workers = _rest_get(f"http://127.0.0.1:{rest_port}/api/workers")
    return {w["id"]: w for w in workers}


def _start_worker(tmp_path: Path, worker_id: str, grpc_port: int, listen_port: int, heartbeat_interval: float) -> subprocess.Popen:
    config = tmp_path / f"{worker_id}.yaml"
    config.write_text(
        f'worker_id: "{worker_id}"\n'
        f'orchestrator_address: "127.0.0.1:{grpc_port}"\n'
        'listen_host: "127.0.0.1"\n'
        f'listen_port: {listen_port}\n'
        f'heartbeat_interval_seconds: {heartbeat_interval}\n'
        'log_level: "info"\n'
    )
    return subprocess.Popen(
        [sys.executable, "-m", "worker.main", "--config", str(config)],
        cwd=str(WORKER_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def test_three_workers_one_fails_two_survive_and_it_reconnects(tmp_path):
    rest_port = 18091
    grpc_port = 19191
    heartbeat_timeout = 2.0  # seconds; short so the test runs in a reasonable time
    heartbeat_interval = 0.5

    orch_config = tmp_path / "orchestrator.yaml"
    orch_config.write_text(
        f'rest_address: "127.0.0.1:{rest_port}"\n'
        f'grpc_address: "127.0.0.1:{grpc_port}"\n'
        'log_level: "info"\n'
        f'heartbeat_timeout_seconds: {heartbeat_timeout}\n'
    )

    orch_proc = subprocess.Popen(
        [str(ORCH_EXE), "--config", str(orch_config)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    worker_procs: dict[str, subprocess.Popen] = {}
    try:
        assert _wait_for(
            lambda: _rest_get(f"http://127.0.0.1:{rest_port}/health")["status"] == "ok",
            timeout=5,
        ), "orchestrator did not become healthy in time"

        ids = ["mw-worker-1", "mw-worker-2", "mw-worker-3"]
        ports = [19210, 19211, 19212]
        for worker_id, port in zip(ids, ports):
            worker_procs[worker_id] = _start_worker(tmp_path, worker_id, grpc_port, port, heartbeat_interval)

        assert _wait_for(
            lambda: set(ids).issubset(_workers_by_id(rest_port).keys()),
            timeout=15,
        ), "not all three workers registered in time"

        by_id = _workers_by_id(rest_port)
        for worker_id in ids:
            assert by_id[worker_id]["online"] is True, f"{worker_id} should start online"

        # Simulate an UNEXPECTED disconnection (hard kill, no
        # Deregister call) of worker 2.
        victim = "mw-worker-2"
        worker_procs[victim].kill()
        worker_procs[victim].wait(timeout=5)

        def victim_offline_others_online() -> bool:
            current = _workers_by_id(rest_port)
            if current.get(victim, {}).get("online", True):
                return False
            return all(current.get(w, {}).get("online") for w in ids if w != victim)

        assert _wait_for(victim_offline_others_online, timeout=heartbeat_timeout + 10), (
            "killed worker did not go offline within the configured timeout, "
            "or a healthy worker incorrectly went offline too"
        )

        # Still LISTED (offline, not removed) -- staleness is a status.
        assert victim in _workers_by_id(rest_port), "offline worker should still be listed, not deleted"

        # Restart it (fresh port, since the old one may still be
        # closing down) and confirm it can register again.
        worker_procs[victim] = _start_worker(tmp_path, victim, grpc_port, 19213, heartbeat_interval)

        assert _wait_for(
            lambda: _workers_by_id(rest_port).get(victim, {}).get("online") is True,
            timeout=15,
        ), "restarted worker did not successfully re-register"

    finally:
        for proc in worker_procs.values():
            if proc.poll() is None:
                proc.kill()
        for proc in worker_procs.values():
            try:
                proc.wait(timeout=5)
            except Exception:
                pass
        orch_proc.kill()
        orch_proc.wait(timeout=5)
