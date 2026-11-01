"""Phase 3 integration tests against the REAL orchestrator binary and
REAL `python -m worker.main` processes (local, loopback, Computer 1
only -- this is local multi-process testing, NOT physical multi-machine
testing; see docs/DEVELOPMENT_STATUS.md).

Covers: real hardware registration over gRPC and its REST exposure
(including /hardware, /metrics, /resources), simulated workers next to
a real one -- clearly separated -- and rejected when the orchestrator
doesn't allow them, dynamic metric refresh, and a worker re-registering
with a fresh profile after the orchestrator restarts and loses its
in-memory registry.

Skipped if orchestrator.exe has not been built.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ORCH_EXE = REPO_ROOT / "orchestrator" / "orchestrator.exe"
WORKER_DIR = REPO_ROOT / "worker"
PROFILE_B = WORKER_DIR / "profiles" / "simulated" / "profile_b_gpu_worker.yaml"
PROFILE_A = WORKER_DIR / "profiles" / "simulated" / "profile_a_cpu_worker.yaml"

pytestmark = pytest.mark.skipif(
    not ORCH_EXE.exists(),
    reason="orchestrator.exe not built; run `go build -o orchestrator.exe ./cmd/orchestrator` in orchestrator/ first",
)

# Real workers import torch and run the compute test at startup
# (~3 s on Computer 1), so registration waits are generous.
REGISTER_TIMEOUT = 40


def _get(url: str, timeout: float = 5.0):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _wait_for(predicate, timeout: float, interval: float = 0.3):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            result = predicate()
            if result:
                return result
        except Exception:
            pass
        time.sleep(interval)
    return None


class Cluster:
    def __init__(self, tmp_path: Path, rest_port: int, grpc_port: int, allow_simulated: bool):
        self.tmp = tmp_path
        self.rest = f"http://127.0.0.1:{rest_port}"
        self.grpc_port = grpc_port
        self.config = tmp_path / "orchestrator.yaml"
        self.config.write_text(
            f'rest_address: "127.0.0.1:{rest_port}"\n'
            f'grpc_address: "127.0.0.1:{grpc_port}"\n'
            "heartbeat_timeout_seconds: 5\n"
            f"allow_simulated_workers: {'true' if allow_simulated else 'false'}\n"
        )
        self.orch = None
        self.workers: dict[str, subprocess.Popen] = {}

    def start_orchestrator(self):
        self.orch = subprocess.Popen([str(ORCH_EXE), "--config", str(self.config)],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        assert _wait_for(lambda: _get(f"{self.rest}/health")["status"] == "ok", timeout=10), "orchestrator not healthy"

    def stop_orchestrator(self):
        self.orch.kill()
        self.orch.wait(timeout=5)

    def start_worker(self, worker_id: str, listen_port: int, simulated_profile: Path | None = None):
        cfg = self.tmp / f"{worker_id}.yaml"
        lines = [
            f'worker_id: "{worker_id}"',
            f'orchestrator_address: "127.0.0.1:{self.grpc_port}"',
            'listen_host: "127.0.0.1"',
            f"listen_port: {listen_port}",
            "heartbeat_interval_seconds: 0.5",
            "gpu_metrics_interval_seconds: 1",
            "hardware_refresh_interval_seconds: 300",
        ]
        if simulated_profile:
            lines.append(f'simulated_profile: "{simulated_profile.as_posix()}"')
        cfg.write_text("\n".join(lines) + "\n")
        log = open(self.tmp / f"{worker_id}.log", "wb")
        self.workers[worker_id] = subprocess.Popen(
            [sys.executable, "-m", "worker.main", "--config", str(cfg)],
            cwd=str(WORKER_DIR), stdout=log, stderr=subprocess.STDOUT,
        )

    def worker(self, worker_id: str):
        try:
            return _get(f"{self.rest}/api/workers/{worker_id}")
        except urllib.error.HTTPError:
            return None

    def log(self, worker_id: str) -> str:
        return (self.tmp / f"{worker_id}.log").read_text(encoding="utf-8", errors="replace")

    def close(self):
        for proc in self.workers.values():
            if proc.poll() is None:
                proc.kill()
        for proc in self.workers.values():
            try:
                proc.wait(timeout=5)
            except Exception:
                pass
        if self.orch and self.orch.poll() is None:
            self.orch.kill()
            self.orch.wait(timeout=5)


@pytest.fixture
def cluster_factory(tmp_path):
    made = []

    def make(**kwargs):
        c = Cluster(tmp_path, **kwargs)
        made.append(c)
        return c

    yield make
    for c in made:
        c.close()


def test_real_and_simulated_workers_are_profiled_and_kept_separate(cluster_factory):
    c = cluster_factory(rest_port=18095, grpc_port=19195, allow_simulated=True)
    c.start_orchestrator()
    c.start_worker("p3-real", 19230)
    c.start_worker("p3-sim-b", 19231, PROFILE_B)

    assert _wait_for(lambda: c.worker("p3-real"), timeout=REGISTER_TIMEOUT), f"real worker did not register:\n{c.log('p3-real')}"
    # The worker registers first and probes backends (in a child
    # process) afterwards; wait for the follow-up registration.
    real = _wait_for(
        lambda: (w := c.worker("p3-real")) and not w["backend_probe_pending"] and w, timeout=REGISTER_TIMEOUT
    )
    assert real, f"backend probe never completed:\n{c.log('p3-real')}"
    log = c.log("p3-real")
    assert log.index("registered with orchestrator") < log.index("inference backend probe finished")
    sim = _wait_for(lambda: c.worker("p3-sim-b"), timeout=REGISTER_TIMEOUT)
    assert sim, f"simulated worker did not register:\n{c.log('p3-sim-b')}"
    assert sim["backend_probe_pending"] is False  # nothing to probe for a simulated profile

    # --- real worker: measured hardware, schema 3, labelled REAL ---
    assert real["profile_source"] == "REAL" and "profile_notice" not in real
    assert real["hardware_schema_version"] == 3
    assert 0 < real["cpu_physical_cores"] <= real["cpu_logical_cores"]
    assert real["profile_collected_at"]
    assert any(b["name"] == "cpu" and b["usable"] for b in real["backend_capabilities"])
    if real["cpu_model"] == "13th Gen Intel(R) Core(TM) i7-1355U":  # Computer 1 ground truth
        assert (real["cpu_physical_cores"], real["cpu_logical_cores"]) == (10, 12)
        assert real["cuda_available"] is False
        (iris,) = [g for g in real["gpus"] if "Iris" in g["name"]]
        assert iris["memory_type"] == "shared" and iris["integrated"] is True
        assert "memory_total_bytes" not in iris and iris["backends"] == []
        assert real["classification"]["classes"] == ["CPU_ONLY"]

    # --- simulated worker: profile values, labelled SIMULATED ---
    assert sim["profile_source"] == "SIMULATED"
    assert sim["profile_notice"] == "SIMULATED - NOT PHYSICALLY VERIFIED"
    assert sim["simulated_profile_name"] == "profile-b-gpu-worker"
    assert sim["hostname"] == "sim-profile-b"
    assert sim["total_ram_bytes"] == 32 * 1024**3
    assert "GPU_ACCELERATED" in sim["classification"]["classes"]
    assert sim["gpus"][0]["dedicated_memory_bytes"] == 8 * 1024**3

    health = _get(f"{c.rest}/health")
    assert health["online_real_workers"] == 1 and health["online_simulated_workers"] == 1

    only_real = _get(f"{c.rest}/api/workers?source=real")
    assert [w["id"] for w in only_real] == ["p3-real"]

    resources = _get(f"{c.rest}/api/resources")
    assert [r["worker_id"] for r in resources["real"]] == ["p3-real"]
    assert [r["worker_id"] for r in resources["simulated"]] == ["p3-sim-b"]
    real_res = resources["real"][0]
    assert real_res["ram_usable_bytes"] < real_res["ram_available_bytes"] <= real_res["ram_total_bytes"]
    assert "cpu" in real_res["execution_backends"]

    hw = _get(f"{c.rest}/api/workers/p3-real/hardware")
    assert hw["cpu"]["physical_cores"] == real["cpu_physical_cores"]

    # --- dynamic metrics keep refreshing (heartbeats every 0.5 s) ---
    first = _get(f"{c.rest}/api/workers/p3-real/metrics")
    assert "available_bytes" in first["ram"]
    later = _wait_for(
        lambda: (m := _get(f"{c.rest}/api/workers/p3-real/metrics"))["metrics_updated_at"] != first["metrics_updated_at"] and m,
        timeout=10,
    )
    assert later, "live metrics did not refresh"
    assert later["metrics_age_seconds"] < 5


def test_simulated_worker_rejected_unless_allowed(cluster_factory):
    c = cluster_factory(rest_port=18096, grpc_port=19196, allow_simulated=False)
    c.start_orchestrator()
    c.start_worker("p3-sim-a", 19232, PROFILE_A)

    proc = c.workers["p3-sim-a"]
    # A rejected registration is final (not retried): the process exits.
    assert _wait_for(lambda: proc.poll() is not None, timeout=REGISTER_TIMEOUT), "rejected worker kept running"
    assert "does not accept SIMULATED" in c.log("p3-sim-a")
    assert c.worker("p3-sim-a") is None
    assert _get(f"{c.rest}/health")["registered_workers"] == 0


def test_worker_reregisters_with_fresh_profile_after_orchestrator_restart(cluster_factory):
    c = cluster_factory(rest_port=18097, grpc_port=19197, allow_simulated=False)
    c.start_orchestrator()
    c.start_worker("p3-reconnect", 19233)
    first = _wait_for(lambda: c.worker("p3-reconnect"), timeout=REGISTER_TIMEOUT)
    assert first, f"worker did not register:\n{c.log('p3-reconnect')}"

    # Orchestrator dies and comes back with an EMPTY in-memory registry.
    c.stop_orchestrator()
    time.sleep(2)  # a few heartbeats fail meanwhile
    c.start_orchestrator()

    again = _wait_for(lambda: c.worker("p3-reconnect"), timeout=REGISTER_TIMEOUT)
    assert again, f"worker never re-registered after orchestrator restart:\n{c.log('p3-reconnect')}"
    assert again["online"] is True
    assert again["profile_source"] == "REAL"
    # A genuinely new profile was collected, not the old one resent.
    assert again["profile_collected_at"] > first["profile_collected_at"]
    assert "re-registered with fresh hardware profile" in c.log("p3-reconnect")
