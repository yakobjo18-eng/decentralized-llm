"""Worker resource sources: simulated profiles, orchestrator JSON, the
live orchestrator (parity with the Go derivation), and this machine."""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request

import pytest

from conftest import PROFILES, REPO
from partitioner.workers import GIB, Phase3Margins, from_local_machine, from_orchestrator, from_resources_json, from_simulated_profile

ORCH_EXE = REPO / "orchestrator" / "orchestrator.exe"


def test_simulated_profile_a_cpu_only():
    w = from_simulated_profile(str(PROFILES / "profile_a_cpu_worker.yaml"))
    assert w.worker_id == "profile-a-cpu-worker" and w.profile_source == "SIMULATED"
    assert w.execution_backends == ("cpu",) and w.gpus == ()
    assert w.ram_total_bytes == 16 * GIB and w.ram_available_bytes == 11 * GIB
    assert w.ram_usable_bytes == 11 * GIB - int(1.6 * GIB)  # 10% of 16 GiB > 1 GiB floor
    assert (w.cpu_physical_cores, w.cpu_logical_cores) == (8, 16)


def test_simulated_profile_b_gpu_vram_usable():
    w = from_simulated_profile(str(PROFILES / "profile_b_gpu_worker.yaml"))
    assert w.execution_backends == ("cpu", "cuda")
    (g,) = w.gpus
    assert g.compute_usable and g.vram_total_bytes == 8 * GIB and g.vram_used_bytes == int(0.6 * GIB)
    assert g.vram_usable_bytes == 8 * GIB - int(0.6 * GIB) - int(0.8 * GIB)


def test_simulated_profile_c_integrated_gpu_has_no_vram():
    w = from_simulated_profile(str(PROFILES / "profile_c_high_memory_worker.yaml"))
    (g,) = w.gpus
    assert not g.compute_usable and g.vram_total_bytes is None and g.vram_usable_bytes is None


def test_resources_json_parsing_keeps_unknowns():
    w = from_resources_json({"worker_id": "x", "profile_source": "REAL", "online": True, "ram_total_bytes": 100,
                             "ram_safety_margin_bytes": 10, "cpu_physical_cores": 2, "cpu_logical_cores": 4,
                             "execution_backends": ["cpu"], "backend_gflops": {"cpu": 31.05}, "gpus": []})
    assert w.ram_available_bytes is None and w.ram_usable_bytes is None  # never filled in from total
    assert w.gflops == {"cpu": 31.05}


def test_margins_match_phase3_defaults():
    m = Phase3Margins()
    assert m.ram(16876888064) == 1687688806  # Computer 1, as the orchestrator reported it
    assert m.ram(4 * GIB) == GIB
    assert m.vram(8 * GIB) == int(0.8 * GIB) and m.vram(2 * GIB) == GIB // 2


@pytest.mark.skipif(not ORCH_EXE.exists(), reason="orchestrator.exe not built")
def test_parity_with_live_orchestrator(tmp_path):
    """The offline derivation equals what the real Go orchestrator
    computes for the same simulated profile."""
    rest, grpc_port = 18089, 19189
    (tmp_path / "o.yaml").write_text(f'rest_address: "127.0.0.1:{rest}"\ngrpc_address: "127.0.0.1:{grpc_port}"\n'
                                     "allow_simulated_workers: true\n")
    profile = PROFILES / "profile_b_gpu_worker.yaml"
    (tmp_path / "w.yaml").write_text(f'worker_id: "parity-b"\norchestrator_address: "127.0.0.1:{grpc_port}"\n'
                                     f'listen_host: "127.0.0.1"\nlisten_port: 19290\nsimulated_profile: "{profile.as_posix()}"\n')
    orch = subprocess.Popen([str(ORCH_EXE), "--config", str(tmp_path / "o.yaml")], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    worker = subprocess.Popen([sys.executable, "-m", "worker.main", "--config", str(tmp_path / "w.yaml")],
                              cwd=str(REPO / "worker"), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        live = None
        deadline = time.time() + 30
        while time.time() < deadline and live is None:
            try:
                live = next((w for w in from_orchestrator(f"http://127.0.0.1:{rest}") if w.worker_id == "parity-b"), None)
            except Exception:
                pass
            time.sleep(0.5)
        assert live is not None, "simulated worker never appeared"
        offline = from_simulated_profile(str(profile), worker_id="parity-b")
        assert live.profile_source == offline.profile_source == "SIMULATED"
        assert live.hostname == offline.hostname == "sim-profile-b"
        for field in ("ram_total_bytes", "ram_available_bytes", "ram_safety_margin_bytes", "ram_usable_bytes",
                      "cpu_physical_cores", "cpu_logical_cores", "execution_backends"):
            assert getattr(live, field) == getattr(offline, field), field
        (lg,), (og,) = live.gpus, offline.gpus
        for field in ("compute_usable", "vram_total_bytes", "vram_used_bytes", "vram_usable_bytes"):
            assert getattr(lg, field) == getattr(og, field), field
    finally:
        worker.kill()
        orch.kill()
        worker.wait(timeout=5)
        orch.wait(timeout=5)


def test_local_machine_is_real_and_conservative():
    """Profiles THIS machine (Computer 1 when run here). Slow (~10 s:
    the Phase 3 backend probe runs in a child process)."""
    w = from_local_machine("computer-1-test")
    assert w.profile_source == "REAL" and w.simulated_profile_name == ""
    assert w.ram_usable_bytes is not None and w.ram_usable_bytes < w.ram_available_bytes <= w.ram_total_bytes
    assert w.cpu_physical_cores <= w.cpu_logical_cores
    assert "cpu" in w.execution_backends
    if w.cpu_logical_cores == 12 and w.ram_total_bytes == 16876888064:  # Computer 1 ground truth
        assert w.cpu_physical_cores == 10
        assert "cuda" not in w.execution_backends
        assert all(not g.compute_usable for g in w.gpus)  # Iris Xe: no compute backend
