"""Simulated hardware profile tests: the shipped examples load, every
simulated snapshot is unmistakably SIMULATED, invalid profiles are
rejected with every problem listed, and a simulated worker never
inspects the real machine."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from worker.communication.registry_client import _build_hardware_info
from worker.generated import common_pb2
from worker.hardware import profiler, simulated
from worker.hardware.simulated import SIMULATED_NOTICE, SimulatedProfileError, load_profile, parse_profile

PROFILE_DIR = Path(__file__).resolve().parents[1] / "profiles" / "simulated"
GIB = 1024**3

VALID = {
    "profile_format_version": 1,
    "profile_source": "SIMULATED",
    "name": "test-worker",
    "cpu": {"model": "Sim CPU", "physical_cores": 4, "logical_cores": 8},
    "ram": {"total_gib": 16, "swap_total_gib": 4},
    "gpus": [{"vendor": "NVIDIA", "model": "Sim GPU", "memory_type": "dedicated", "vram_gib": 6, "compute_backends": ["cuda"]}],
    "dynamic": {"cpu_utilization_percent": 10, "ram_available_gib": 12, "gpus": [{"vram_used_gib": 1}]},
}


def _with(**changes):
    data = copy.deepcopy(VALID)
    for dotted, value in changes.items():
        target = data
        *path, last = dotted.split("__")
        for key in path:
            target = target[int(key)] if isinstance(target, list) else target[key]
        if value is _DELETE:
            del target[last]
        else:
            target[last] = value
    return data


_DELETE = object()


def test_shipped_example_profiles_are_valid():
    profiles = simulated.load_profiles_dir(str(PROFILE_DIR))
    names = {p.name for p in profiles}
    assert names == {"profile-a-cpu-worker", "profile-b-gpu-worker", "profile-c-high-memory-worker"}
    for p in profiles:
        assert p.snapshot.profile_source == "SIMULATED"
        assert p.snapshot.simulated_profile_name == p.name
        assert p.snapshot.hardware_schema_version == profiler.HARDWARE_SCHEMA_VERSION
        assert p.notice == SIMULATED_NOTICE
        assert p.snapshot.cpu_physical_cores <= p.snapshot.cpu_logical_cores
        for cap in p.snapshot.backend_capabilities:
            # Never claims to have executed anything.
            assert cap.tested is False and cap.test_passed is None
            assert SIMULATED_NOTICE in cap.detail


def test_shipped_example_files_state_they_are_not_computers_2_and_3():
    for path in PROFILE_DIR.glob("*.yaml"):
        text = path.read_text(encoding="utf-8")
        assert text.startswith("# SIMULATED - NOT PHYSICALLY VERIFIED"), path.name
        comments = " ".join(line.lstrip("# ").strip() for line in text.splitlines() if line.startswith("#"))
        assert "NOT the specification of Computer 2 or Computer 3" in " ".join(comments.split()), path.name


def test_profile_a_cpu_only():
    p = load_profile(str(PROFILE_DIR / "profile_a_cpu_worker.yaml"))
    assert p.snapshot.gpus == []
    assert p.snapshot.cuda_available is False
    assert [c.name for c in p.snapshot.backend_capabilities] == ["cpu"]


def test_profile_b_gpu_values_from_file():
    p = load_profile(str(PROFILE_DIR / "profile_b_gpu_worker.yaml"))
    (gpu,) = p.snapshot.gpus
    assert gpu.vendor == "NVIDIA" and gpu.memory_type == "dedicated"
    assert gpu.dedicated_memory_bytes == 8 * GIB
    assert p.snapshot.cuda_available is True
    assert p.live.gpu_metrics[0].gpu_id == gpu.id
    assert p.live.gpu_metrics[0].dedicated_memory_used_bytes == int(0.6 * GIB)


def test_profile_c_integrated_gpu_has_no_vram_and_no_compute():
    p = load_profile(str(PROFILE_DIR / "profile_c_high_memory_worker.yaml"))
    (gpu,) = p.snapshot.gpus
    assert gpu.memory_type == "shared" and gpu.integrated
    assert gpu.memory_total_bytes is None and gpu.dedicated_memory_bytes is None
    assert gpu.backends == []
    assert p.snapshot.total_ram_bytes == 64 * GIB


def test_configurable_values_flow_through():
    p = parse_profile(_with(cpu__physical_cores=6, cpu__logical_cores=6, ram__total_gib=48, gpus__0__vram_gib=24))
    assert p.snapshot.cpu_physical_cores == 6
    assert p.snapshot.total_ram_bytes == 48 * GIB
    assert p.snapshot.gpus[0].memory_total_bytes == 24 * GIB
    assert p.live.ram_available_bytes == 12 * GIB
    assert p.live.ram_used_bytes == 36 * GIB


@pytest.mark.parametrize(
    "changes, expected",
    [
        ({"profile_source": "REAL"}, "profile_source must be exactly 'SIMULATED'"),
        ({"profile_source": _DELETE}, "profile_source must be exactly"),
        ({"profile_format_version": 2}, "profile_format_version must be 1"),
        ({"name": "Bad Name!"}, "lowercase letters"),
        ({"cpu__physical_cores": 16}, "cannot exceed cpu.logical_cores"),
        ({"cpu__physical_cores": 0}, "physical_cores must be >= 1"),
        ({"cpu__logical_cores": 7.5}, "whole number"),
        ({"cpu__model": _DELETE}, "cpu.model is required"),
        ({"cpu": _DELETE}, "cpu section is required"),
        ({"ram__total_gib": -1}, "total_gib must be > 0"),
        ({"ram__total_gib": "lots"}, "must be a number"),
        ({"dynamic__ram_available_gib": 99}, "cannot exceed ram.total_gib"),
        ({"dynamic__cpu_utilization_percent": 140}, "must be <= 100"),
        ({"gpus__0__vendor": "Matrox"}, "vendor must be one of"),
        ({"gpus__0__memory_type": "shared"}, "cannot have vram_gib"),
        ({"gpus__0__vram_gib": _DELETE}, "vram_gib is required"),
        ({"gpus__0__compute_backends": ["cuda", "warp-drive"]}, "unknown backend 'warp-drive'"),
        ({"gpus__0__vendor": "Intel"}, "'cuda' is not available on Intel GPUs"),
        ({"dynamic__gpus": [{"vram_used_gib": 7}]}, "exceeds gpus[0]'s vram_gib"),
        ({"dynamic__gpus": [{}, {}]}, "only 1 gpus are declared"),
        ({"cpu__cores": 8}, "unknown key 'cores'"),
        ({"extra_section": {}}, "unknown key 'extra_section'"),
    ],
)
def test_invalid_profiles_rejected(changes, expected):
    with pytest.raises(SimulatedProfileError, match=None) as exc:
        parse_profile(_with(**changes), source="test")
    assert expected in str(exc.value)


def test_all_problems_reported_at_once():
    bad = _with(**{"profile_source": "REAL", "cpu__physical_cores": 99, "ram__total_gib": 0})
    with pytest.raises(SimulatedProfileError) as exc:
        parse_profile(bad)
    message = str(exc.value)
    assert "profile_source" in message and "physical_cores" in message and "total_gib" in message


def test_load_profile_errors(tmp_path):
    with pytest.raises(SimulatedProfileError, match="not found"):
        load_profile(str(tmp_path / "missing.yaml"))
    broken = tmp_path / "broken.yaml"
    broken.write_text("cpu: [unclosed", encoding="utf-8")
    with pytest.raises(SimulatedProfileError, match="not valid YAML"):
        load_profile(str(broken))
    (tmp_path / "a.yaml").write_text(yaml.safe_dump(VALID), encoding="utf-8")
    (tmp_path / "b.yaml").write_text(yaml.safe_dump(VALID), encoding="utf-8")
    broken.unlink()
    with pytest.raises(SimulatedProfileError, match="duplicate"):
        simulated.load_profiles_dir(str(tmp_path))


def test_simulated_profile_goes_over_the_wire_as_simulated():
    p = parse_profile(VALID)
    info = _build_hardware_info(p.snapshot)
    assert info.profile_source == common_pb2.PROFILE_SOURCE_SIMULATED
    assert info.simulated_profile_name == "test-worker"
    assert info.hostname == "sim-test-worker"
    assert info.gpus[0].memory_source == "simulated"


def test_simulated_profiler_never_touches_real_hardware(monkeypatch):
    def _fail(*a, **k):
        raise AssertionError("a simulated worker must not inspect the real machine")

    monkeypatch.setattr(profiler, "collect", _fail)
    monkeypatch.setattr(profiler, "collect_live_metrics", _fail)
    monkeypatch.setattr(profiler, "_detect_gpus", _fail)
    p = parse_profile(VALID)
    hp = profiler.HardwareProfiler(simulated_snapshot=p.snapshot, simulated_live=p.live)
    assert hp.is_simulated
    snap = hp.full_snapshot()
    assert snap.profile_source == "SIMULATED"
    assert snap.profile_collected_at_unix_ms > 0
    live = hp.live_metrics()
    assert live.cpu_utilization_percent == 10
    assert live.gpu_metrics and live.gpu_metrics[0].gpu_id == "sim:test-worker:0"


def test_cli_validate(capsys):
    assert simulated._main(["validate", *map(str, sorted(PROFILE_DIR.glob("*.yaml")))]) == 0
    out = capsys.readouterr().out
    assert out.count("OK") == 3 and SIMULATED_NOTICE in out
