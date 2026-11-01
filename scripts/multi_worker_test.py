"""Phase 2 manual multi-worker runner.

Starts the real orchestrator and three real worker processes (each a
separate OS process, its own worker_id, port, and config file), prints
the live registry state from the REST API every few seconds, and tears
everything down cleanly on Ctrl+C (gracefully, via CTRL_BREAK_EVENT on
Windows, so you can see deregistration happen in the printed state).

This is the interactive, watch-it-happen counterpart to the automated
pytest integration test (worker/tests/test_three_worker_integration.py),
which asserts the same scenario without a human watching. Use this one
to actually see three workers appear, kill one yourself in another
terminal (e.g. `Stop-Process -Id <pid> -Force`, where the PID for each
worker is printed below), watch it go offline, and watch it come back
if you restart it.

Usage (from the repository root, orchestrator.exe already built):

    python scripts/multi_worker_test.py

Requires: orchestrator/orchestrator.exe built
(`go build -o orchestrator.exe ./cmd/orchestrator` in orchestrator/).
"""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ORCH_EXE = REPO_ROOT / "orchestrator" / "orchestrator.exe"
WORKER_DIR = REPO_ROOT / "worker"

REST_PORT = 18095
GRPC_PORT = 19195
HEARTBEAT_TIMEOUT = 8.0
WORKER_SPECS = [
    ("manual-worker-1", 19220),
    ("manual-worker-2", 19221),
    ("manual-worker-3", 19222),
]


def _rest_get(url: str, timeout: float = 5.0):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _write_orchestrator_config(path: Path) -> None:
    path.write_text(
        f'rest_address: "127.0.0.1:{REST_PORT}"\n'
        f'grpc_address: "127.0.0.1:{GRPC_PORT}"\n'
        'log_level: "info"\n'
        f'heartbeat_timeout_seconds: {HEARTBEAT_TIMEOUT}\n'
    )


def _write_worker_config(path: Path, worker_id: str, port: int) -> None:
    path.write_text(
        f'worker_id: "{worker_id}"\n'
        f'orchestrator_address: "127.0.0.1:{GRPC_PORT}"\n'
        'listen_host: "127.0.0.1"\n'
        f'listen_port: {port}\n'
        'heartbeat_interval_seconds: 2.0\n'
        'log_level: "info"\n'
    )


def main() -> int:
    if not ORCH_EXE.exists():
        print(f"ERROR: {ORCH_EXE} not found. Build it first:")
        print("    cd orchestrator; go build -o orchestrator.exe ./cmd/orchestrator")
        return 1

    run_dir = REPO_ROOT / "scripts" / ".multi_worker_run"
    run_dir.mkdir(parents=True, exist_ok=True)

    orch_config = run_dir / "orchestrator.yaml"
    _write_orchestrator_config(orch_config)

    print(f"Starting orchestrator on REST=127.0.0.1:{REST_PORT} gRPC=127.0.0.1:{GRPC_PORT} "
          f"(heartbeat_timeout={HEARTBEAT_TIMEOUT}s)...")
    orch_proc = subprocess.Popen([str(ORCH_EXE), "--config", str(orch_config)])
    print(f"  orchestrator PID: {orch_proc.pid}")

    worker_procs: dict[str, subprocess.Popen] = {}
    try:
        time.sleep(1.5)
        try:
            _rest_get(f"http://127.0.0.1:{REST_PORT}/health")
        except Exception as exc:
            print(f"ERROR: orchestrator did not come up: {exc}")
            return 1

        for worker_id, port in WORKER_SPECS:
            config_path = run_dir / f"{worker_id}.yaml"
            _write_worker_config(config_path, worker_id, port)
            proc = subprocess.Popen(
                [sys.executable, "-m", "worker.main", "--config", str(config_path)],
                cwd=str(WORKER_DIR),
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
            )
            worker_procs[worker_id] = proc
            print(f"Started {worker_id} (PID {proc.pid}, port {port})")

        print("\nAll three workers launched. Printing live registry state every 3s.")
        print("Press Ctrl+C to shut everything down gracefully.\n")

        while True:
            time.sleep(3)
            try:
                workers = _rest_get(f"http://127.0.0.1:{REST_PORT}/api/workers")
            except Exception as exc:
                print(f"  (orchestrator unreachable: {exc})")
                continue
            print(f"--- {time.strftime('%H:%M:%S')} ---")
            for w in workers:
                print(
                    f"  {w['id']:<18} online={w['online']!s:<5} "
                    f"status={w['status']:<6} last_heartbeat_age={w['last_heartbeat_age_seconds']:.1f}s"
                )
            if not workers:
                print("  (no workers registered)")

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        for worker_id, proc in worker_procs.items():
            if proc.poll() is None:
                if sys.platform == "win32":
                    try:
                        proc.send_signal(signal.CTRL_BREAK_EVENT)
                    except Exception:
                        proc.terminate()
                else:
                    proc.terminate()
        for worker_id, proc in worker_procs.items():
            try:
                proc.wait(timeout=5)
                print(f"{worker_id} stopped (deregistered).")
            except subprocess.TimeoutExpired:
                print(f"{worker_id} did not stop gracefully in time, killing.")
                proc.kill()

        orch_proc.terminate()
        try:
            orch_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            orch_proc.kill()
        print("Orchestrator stopped.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
