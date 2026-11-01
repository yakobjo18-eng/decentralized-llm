"""Phase 2 integration test: starts the REAL orchestrator.exe and a
REAL `python -m worker.main` subprocess, sends it a graceful
CTRL_BREAK_EVENT (Windows' equivalent of Ctrl+C targeted at one
specific background process), and confirms via the orchestrator's live
REST API that the worker deregistered itself.

This specifically closes a gap noted in Phase 1: the Phase 1 smoke
test could only force-kill the worker (Stop-Process), which never
exercises the graceful-shutdown/deregister code path at all. Using
CREATE_NEW_PROCESS_GROUP + CTRL_BREAK_EVENT lets this test target one
specific worker process without killing anything else sharing this
console.

Skipped if orchestrator.exe has not been built, or on a non-Windows
platform (CTRL_BREAK_EVENT is Windows-specific).
"""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ORCH_EXE = REPO_ROOT / "orchestrator" / "orchestrator.exe"
WORKER_DIR = REPO_ROOT / "worker"

pytestmark = [
    pytest.mark.skipif(sys.platform != "win32", reason="CTRL_BREAK_EVENT is Windows-specific"),
    pytest.mark.skipif(
        not ORCH_EXE.exists(),
        reason="orchestrator.exe not built; run `go build -o orchestrator.exe ./cmd/orchestrator` in orchestrator/ first",
    ),
]


def _rest_get(url: str, timeout: float = 5.0):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _wait_for(predicate, timeout: float, interval: float = 0.2) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if predicate():
                return True
        except Exception:
            pass
        time.sleep(interval)
    return False


def test_worker_deregisters_on_graceful_ctrl_break(tmp_path):
    rest_port = 18090
    grpc_port = 19190
    worker_port = 19200
    worker_id = "pytest-graceful-worker"

    orch_config = tmp_path / "orchestrator.yaml"
    orch_config.write_text(
        f'rest_address: "127.0.0.1:{rest_port}"\n'
        f'grpc_address: "127.0.0.1:{grpc_port}"\n'
        'log_level: "info"\n'
    )

    worker_config = tmp_path / "worker.yaml"
    worker_config.write_text(
        f'worker_id: "{worker_id}"\n'
        f'orchestrator_address: "127.0.0.1:{grpc_port}"\n'
        'listen_host: "127.0.0.1"\n'
        f'listen_port: {worker_port}\n'
        'heartbeat_interval_seconds: 1.0\n'
        'log_level: "info"\n'
    )

    orch_proc = subprocess.Popen(
        [str(ORCH_EXE), "--config", str(orch_config)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        assert _wait_for(
            lambda: _rest_get(f"http://127.0.0.1:{rest_port}/health")["status"] == "ok",
            timeout=5,
        ), "orchestrator did not become healthy in time"

        worker_proc = subprocess.Popen(
            [sys.executable, "-m", "worker.main", "--config", str(worker_config)],
            cwd=str(WORKER_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
        try:

            def _worker_listed() -> bool:
                workers = _rest_get(f"http://127.0.0.1:{rest_port}/api/workers")
                return any(w["id"] == worker_id for w in workers)

            assert _wait_for(_worker_listed, timeout=15), "worker did not register in time"

            # Graceful shutdown of THIS worker only.
            worker_proc.send_signal(signal.CTRL_BREAK_EVENT)

            assert _wait_for(lambda: not _worker_listed(), timeout=10), (
                "worker was not deregistered after a graceful CTRL_BREAK_EVENT shutdown "
                "-- deregister-on-shutdown is not actually working"
            )
        finally:
            if worker_proc.poll() is None:
                worker_proc.kill()
                worker_proc.wait(timeout=5)
    finally:
        orch_proc.kill()
        orch_proc.wait(timeout=5)
