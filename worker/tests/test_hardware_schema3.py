"""Phase 3 (hardware schema 3) unit tests: CPU frequency/load, RAM and
swap, DXGI-based GPU memory classification for NVIDIA / AMD / Intel,
PDH counter aggregation, inference backend probing with the verified
compute test, profile source/versioning, and the HardwareProfiler's
static-vs-dynamic caching.

Hardware this project doesn't have (NVIDIA, AMD, Intel Arc, working
CUDA) is simulated by monkeypatching the detection sources. Tests that
assert facts about Computer 1 itself are marked "ground truth" and
skip on any other machine instead of failing.
"""

from __future__ import annotations

import platform
import subprocess
import sys
import time
import types

import pytest

from worker.communication.registry_client import _build_hardware_info
from worker.generated import common_pb2
from worker.hardware import backend_probe, profiler, windows_gpu
from worker.hardware.profiler import GPUInfo, HardwareSnapshot, LiveMetrics

GIB = 1024**3


def _adapter(name, vendor_id, dedicated, shared=8 * GIB, luid="luid:0x00000000_0x00001111", software=False):
    return windows_gpu.DXGIAdapter(
        description=name,
        vendor_id=vendor_id,
        device_id=1,
        dedicated_video_memory=dedicated,
        dedicated_system_memory=0,
        shared_system_memory=shared,
        luid=luid,
        is_software=software,
    )


def _wmi(name, pnp, adapter_ram=0, driver="1.0"):
    return {"Name": name, "AdapterRAM": adapter_ram, "DriverVersion": driver, "PNPDeviceID": pnp}


@pytest.fixture
def no_gpu_sources(monkeypatch):
    monkeypatch.setattr(profiler, "_detect_nvidia_gpus_via_smi", lambda: [])
    monkeypatch.setattr(profiler, "_detect_gpus_via_powershell", lambda: [])
    monkeypatch.setattr(profiler, "_detect_gpus_via_dxgi", lambda: [])


def _fake_psutil(monkeypatch, **attrs):
    fake = types.SimpleNamespace(**attrs)
    monkeypatch.setitem(sys.modules, "psutil", fake)
    return fake


# ---------------------------------------------------------------------
# CPU
# ---------------------------------------------------------------------


def test_cpu_identity_ground_truth_computer_1():
    manufacturer, model = profiler._cpu_identity()
    if "i7-1355U" not in model:
        pytest.skip(f"not Computer 1 (cpu model {model!r})")
    assert manufacturer == "Intel"
    assert model == "13th Gen Intel(R) Core(TM) i7-1355U"


def test_cpu_identity_falls_back_to_platform_when_registry_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "winreg", None)  # import fails
    monkeypatch.setattr(profiler.platform, "processor", lambda: "AMD64 Family 25 Model 33, AuthenticAMD")
    manufacturer, model = profiler._cpu_identity()
    assert manufacturer == "AMD"
    assert model == "AMD64 Family 25 Model 33, AuthenticAMD"


def test_cpu_identity_unknown_when_nothing_available(monkeypatch):
    monkeypatch.setitem(sys.modules, "winreg", None)
    monkeypatch.setattr(profiler.platform, "processor", lambda: "")
    assert profiler._cpu_identity() == ("Unknown", "Unknown")


def test_physical_core_count_unknown_is_zero_not_logical(monkeypatch):
    """psutil returns None for physical cores on some platforms. That
    must become 0 ("unknown"), never silently the logical count."""
    _fake_psutil(monkeypatch, cpu_count=lambda logical=True: 12 if logical else None)
    assert profiler._cpu_counts() == (0, 12)


def test_cpu_counts_survive_psutil_failure(monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", None)
    physical, logical = profiler._cpu_counts()
    assert physical == 0
    assert logical == (profiler.os.cpu_count() or 0)


def test_cpu_frequency_zero_means_unknown(monkeypatch):
    _fake_psutil(monkeypatch, cpu_freq=lambda: types.SimpleNamespace(current=0.0, min=0.0, max=0.0))
    assert profiler._cpu_frequency_mhz() == (None, None)


def test_cpu_frequency_missing_entirely(monkeypatch):
    _fake_psutil(monkeypatch, cpu_freq=lambda: None)
    assert profiler._cpu_frequency_mhz() == (None, None)

    def _boom():
        raise OSError("no frequency info")

    _fake_psutil(monkeypatch, cpu_freq=_boom)
    assert profiler._cpu_frequency_mhz() == (None, None)


def test_cpu_utilization_unavailable_is_none(monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", None)
    assert profiler._cpu_utilization_percent() is None


def test_load_average_hidden_during_windows_warmup(monkeypatch):
    _fake_psutil(monkeypatch, getloadavg=lambda: (0.0, 0.0, 0.0))
    monkeypatch.setattr(profiler.os, "name", "nt")
    monkeypatch.setattr(profiler, "_load_avg_started_at", None)
    # psutil's Windows emulation returns zeros at first; those zeros
    # are not a measurement and must not be reported as one.
    assert profiler._cpu_load_average_1m() is None

    monkeypatch.setattr(profiler, "_load_avg_started_at", time.monotonic() - 120)
    _fake_psutil(monkeypatch, getloadavg=lambda: (1.25, 0.9, 0.5))
    assert profiler._cpu_load_average_1m() == 1.25


# ---------------------------------------------------------------------
# RAM / swap
# ---------------------------------------------------------------------


def test_ram_snapshot_is_consistent_on_this_machine():
    total, available, used, pct = profiler._ram_snapshot()
    assert total > 0
    assert 0 < available <= total
    assert 0 < used <= total
    assert 0 <= pct <= 100


def test_ram_ground_truth_computer_1():
    total, _, _, _ = profiler._ram_snapshot()
    _, logical = profiler._cpu_counts()
    if logical != 12 or "i7-1355U" not in profiler._cpu_identity()[1]:
        pytest.skip("not Computer 1")
    # Previously verified: ~15.72 GB usable physical RAM.
    assert 15.5e9 < total < 17.5e9


def test_ram_unavailable_returns_unknown_not_zero(monkeypatch):
    def _boom():
        raise RuntimeError("psutil broken")

    _fake_psutil(monkeypatch, virtual_memory=_boom)
    assert profiler._ram_snapshot() == (0, None, None, None)


def test_swap_unavailable_returns_none(monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", None)
    assert profiler._swap_snapshot() == (None, None, None)


def test_live_metrics_survive_total_psutil_failure(monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", None)
    live = profiler.collect_live_metrics()
    assert live.cpu_utilization_percent is None
    assert live.ram_available_bytes is None
    assert live.swap_used_bytes is None
    assert live.collected_at_unix_ms > 0


# ---------------------------------------------------------------------
# GPU: NVIDIA / AMD / Intel, DXGI enrichment, detection failure != absence
# ---------------------------------------------------------------------


def test_nvidia_smi_output_parsed(monkeypatch):
    monkeypatch.setattr(profiler.shutil, "which", lambda name: "nvidia-smi")
    monkeypatch.setattr(
        profiler.subprocess,
        "check_output",
        lambda *a, **k: b"NVIDIA GeForce RTX 3060, 12288, 551.23\nNVIDIA RTX A2000, [N/A], 551.23\n",
    )
    gpus = profiler._detect_nvidia_gpus_via_smi()
    assert [g.name for g in gpus] == ["NVIDIA GeForce RTX 3060", "NVIDIA RTX A2000"]
    assert gpus[0].memory_total_bytes == 12 * GIB
    assert gpus[0].backends == ["CUDA"] and gpus[0].memory_type == "dedicated"
    assert gpus[1].memory_total_bytes is None  # "[N/A]" is unknown, not 0


def test_nvidia_detected_with_dxgi_keeps_nvidia_smi_vram(monkeypatch):
    monkeypatch.setattr(
        profiler,
        "_detect_nvidia_gpus_via_smi",
        lambda: [GPUInfo("NVIDIA", "NVIDIA GeForce RTX 3060", 12 * GIB, "dedicated", ["CUDA"], "551.23")],
    )
    monkeypatch.setattr(profiler, "_detect_gpus_via_powershell", lambda: [_wmi("NVIDIA GeForce RTX 3060", r"PCI\VEN_10DE&DEV_2504", 4294967295)])
    monkeypatch.setattr(
        profiler, "_detect_gpus_via_dxgi", lambda: [_adapter("NVIDIA GeForce RTX 3060", 0x10DE, 12 * GIB - 1)]
    )
    (gpu,) = profiler._detect_gpus()
    assert gpu.memory_source == "nvidia-smi"
    assert gpu.dedicated_memory_bytes == 12 * GIB  # authoritative, not DXGI's
    assert gpu.id == "luid:0x00000000_0x00001111"
    assert gpu.integrated is False


def test_amd_discrete_gpu_uses_dxgi_64bit_vram(monkeypatch):
    """WMI reports 4 GiB (32-bit overflow) for a 16 GiB card; DXGI's
    64-bit figure must win."""
    monkeypatch.setattr(profiler, "_detect_nvidia_gpus_via_smi", lambda: [])
    monkeypatch.setattr(
        profiler, "_detect_gpus_via_powershell", lambda: [_wmi("AMD Radeon RX 7800 XT", r"PCI\VEN_1002&DEV_747E", 4293918720)]
    )
    monkeypatch.setattr(profiler, "_detect_gpus_via_dxgi", lambda: [_adapter("AMD Radeon RX 7800 XT", 0x1002, 16 * GIB)])
    (gpu,) = profiler._detect_gpus()
    assert gpu.vendor == "AMD"
    assert gpu.memory_type == "dedicated"
    assert gpu.memory_total_bytes == 16 * GIB
    assert gpu.memory_source == "dxgi"
    assert gpu.backends == []  # no compute backend inferred from hardware


def test_amd_apu_is_shared_not_dedicated(monkeypatch):
    monkeypatch.setattr(profiler, "_detect_nvidia_gpus_via_smi", lambda: [])
    monkeypatch.setattr(profiler, "_detect_gpus_via_powershell", lambda: [_wmi("AMD Radeon(TM) Graphics", r"PCI\VEN_1002&DEV_1638", 536870912)])
    monkeypatch.setattr(profiler, "_detect_gpus_via_dxgi", lambda: [_adapter("AMD Radeon(TM) Graphics", 0x1002, 512 * 1024**2)])
    (gpu,) = profiler._detect_gpus()
    assert gpu.memory_type == "shared"
    assert gpu.integrated is True
    assert gpu.memory_total_bytes is None


def test_amd_seen_only_via_wmi_is_unknown_memory_type(monkeypatch):
    monkeypatch.setattr(profiler, "_detect_nvidia_gpus_via_smi", lambda: [])
    monkeypatch.setattr(profiler, "_detect_gpus_via_powershell", lambda: [_wmi("AMD Radeon RX 6600", r"PCI\VEN_1002&DEV_73FF", 4293918720)])
    monkeypatch.setattr(profiler, "_detect_gpus_via_dxgi", lambda: [])
    (gpu,) = profiler._detect_gpus()
    # WMI alone can't distinguish an APU from a discrete card.
    assert gpu.memory_type == "unknown"
    assert gpu.memory_total_bytes is None


def test_intel_iris_xe_shared_memory_never_reported_as_vram(monkeypatch):
    monkeypatch.setattr(profiler, "_detect_nvidia_gpus_via_smi", lambda: [])
    monkeypatch.setattr(
        profiler,
        "_detect_gpus_via_powershell",
        lambda: [_wmi("Intel(R) Iris(R) Xe Graphics", r"PCI\VEN_8086&DEV_A7A1", 2147479552, "32.0.101.6881")],
    )
    monkeypatch.setattr(
        profiler,
        "_detect_gpus_via_dxgi",
        lambda: [_adapter("Intel(R) Iris(R) Xe Graphics", 0x8086, 128 * 1024**2, shared=8438444032)],
    )
    (gpu,) = profiler._detect_gpus()
    assert gpu.vendor == "Intel"
    assert gpu.memory_type == "shared"
    assert gpu.integrated is True
    assert gpu.memory_total_bytes is None  # NOT the 2 GiB WMI aperture
    assert gpu.dedicated_memory_bytes == 128 * 1024**2  # raw DXGI carve-out, documented as system RAM
    assert gpu.shared_memory_bytes == 8438444032
    assert "CUDA" not in gpu.backends
    assert gpu.driver_version == "32.0.101.6881"


def test_intel_arc_discrete_is_dedicated(monkeypatch):
    monkeypatch.setattr(profiler, "_detect_nvidia_gpus_via_smi", lambda: [])
    monkeypatch.setattr(profiler, "_detect_gpus_via_powershell", lambda: [_wmi("Intel(R) Arc(TM) A770 Graphics", r"PCI\VEN_8086&DEV_56A0")])
    monkeypatch.setattr(profiler, "_detect_gpus_via_dxgi", lambda: [_adapter("Intel(R) Arc(TM) A770 Graphics", 0x8086, 16 * GIB)])
    (gpu,) = profiler._detect_gpus()
    assert gpu.memory_type == "dedicated"
    assert gpu.memory_total_bytes == 16 * GIB


def test_intel_arc_branded_integrated_gpu_is_shared(monkeypatch):
    """Meteor Lake's integrated GPU is also called "Intel Arc"; the
    measured DXGI size, not the name, decides."""
    monkeypatch.setattr(profiler, "_detect_nvidia_gpus_via_smi", lambda: [])
    monkeypatch.setattr(profiler, "_detect_gpus_via_powershell", lambda: [_wmi("Intel(R) Arc(TM) Graphics", r"PCI\VEN_8086&DEV_7D55")])
    monkeypatch.setattr(profiler, "_detect_gpus_via_dxgi", lambda: [_adapter("Intel(R) Arc(TM) Graphics", 0x8086, 128 * 1024**2)])
    (gpu,) = profiler._detect_gpus()
    assert gpu.memory_type == "shared" and gpu.memory_total_bytes is None

    # And without DXGI the name alone is not enough to decide.
    monkeypatch.setattr(profiler, "_detect_gpus_via_dxgi", lambda: [])
    (gpu,) = profiler._detect_gpus()
    assert gpu.memory_type == "unknown"


def test_wmi_failure_does_not_mean_no_gpu(monkeypatch):
    """PowerShell missing/failing must not make the GPU disappear:
    DXGI alone still produces the inventory."""
    monkeypatch.setattr(profiler, "_detect_nvidia_gpus_via_smi", lambda: [])
    monkeypatch.setattr(profiler, "_detect_gpus_via_powershell", lambda: [])
    monkeypatch.setattr(profiler, "_detect_gpus_via_dxgi", lambda: [_adapter("Intel(R) UHD Graphics", 0x8086, 128 * 1024**2)])
    (gpu,) = profiler._detect_gpus()
    assert gpu.vendor == "Intel"
    assert gpu.memory_type == "shared"


def test_software_rasterizer_is_not_a_gpu(monkeypatch):
    monkeypatch.setattr(
        profiler.windows_gpu,
        "enumerate_dxgi_adapters",
        lambda: [_adapter("Microsoft Basic Render Driver", 0x1414, 0, software=True)],
    )
    assert profiler._detect_gpus_via_dxgi() == []


def test_unavailable_vram_metric_is_none(monkeypatch):
    """Unknown vendor seen only by WMI with AdapterRAM missing: VRAM
    unknown (None), GPU still reported."""
    monkeypatch.setattr(profiler, "_detect_nvidia_gpus_via_smi", lambda: [])
    monkeypatch.setattr(profiler, "_detect_gpus_via_powershell", lambda: [{"Name": "Mystery Display Adapter", "PNPDeviceID": ""}])
    monkeypatch.setattr(profiler, "_detect_gpus_via_dxgi", lambda: [])
    (gpu,) = profiler._detect_gpus()
    assert gpu.vendor == "Unknown"
    assert gpu.memory_total_bytes is None and gpu.dedicated_memory_bytes is None


def test_gpu_ground_truth_computer_1_iris_xe():
    gpus = profiler._detect_gpus()
    iris = [g for g in gpus if "Iris(R) Xe" in g.name]
    if not iris:
        pytest.skip("no Iris Xe here -- not Computer 1")
    gpu = iris[0]
    assert gpu.vendor == "Intel"
    assert gpu.memory_type == "shared" and gpu.integrated
    assert gpu.memory_total_bytes is None
    assert gpu.backends == []  # not a CUDA GPU
    assert not any(g.vendor == "NVIDIA" for g in gpus)


def test_engine_utilization_aggregation():
    samples = {
        "pid_1_luid_0x00000000_0x000160E5_phys_0_eng_0_engtype_3D": 20.0,
        "pid_2_luid_0x00000000_0x000160E5_phys_0_eng_0_engtype_3D": 15.0,
        "pid_2_luid_0x00000000_0x000160E5_phys_0_eng_3_engtype_VideoDecode": 30.0,
        "pid_3_luid_0x00000000_0x000164AC_phys_0_eng_0_engtype_3D": 90.0,
        "pid_4_luid_0x00000000_0x000164AC_phys_0_eng_0_engtype_3D": 40.0,
        "pid_5_luid_0x00000000_0x000164AC_phys_0_eng_0_engtype_3D": float("nan"),
        "garbage": 50.0,
    }
    out = windows_gpu.aggregate_engine_utilization(samples)
    # busiest engine type per adapter, summed over processes, capped
    assert out["luid:0x00000000_0x000160E5"] == 35.0
    assert out["luid:0x00000000_0x000164AC"] == 100.0
    assert len(out) == 2


def test_adapter_memory_aggregation():
    out = windows_gpu.aggregate_adapter_memory(
        {"luid_0x00000000_0x000160e5_phys_0": 1566130176.0, "luid_0x00000000_0x000164AC_phys_0": 8192.0, "x": 5.0}
    )
    assert out == {"luid:0x00000000_0x000160E5": 1566130176, "luid:0x00000000_0x000164AC": 8192}


def test_gpu_counter_sampler_unsupported_platform(monkeypatch):
    monkeypatch.setattr(windows_gpu.sys, "platform", "linux")
    sampler = windows_gpu.GPUCounterSampler()
    assert sampler.supported is False
    assert sampler.sample() == {}
    assert windows_gpu.enumerate_dxgi_adapters() == []


def test_collect_gpu_metrics_only_reports_known_gpus():
    class FakeSampler:
        def sample(self):
            return {
                "luid:A": windows_gpu.GPUCounterSample(utilization_percent=12.5, shared_memory_used_bytes=100),
                "luid:SOFTWARE": windows_gpu.GPUCounterSample(utilization_percent=1.0),
            }

    gpus = [GPUInfo("Intel", "iGPU", None, "shared", id="luid:A")]
    (metric,) = profiler.collect_gpu_metrics(FakeSampler(), gpus)
    assert metric.gpu_id == "luid:A" and metric.utilization_percent == 12.5
    assert metric.dedicated_memory_used_bytes is None
    assert profiler.collect_gpu_metrics(FakeSampler(), []) == []


# ---------------------------------------------------------------------
# Inference backends
# ---------------------------------------------------------------------


class _FakeCuda:
    def __init__(self, available, count=0, names=()):
        self._available, self._count, self._names = available, count, list(names)

    def is_available(self):
        return self._available

    def device_count(self):
        return self._count

    def get_device_name(self, i):
        return self._names[i]


def _fake_torch(cuda, cuda_build="12.6", xpu=None, mps=None):
    torch = types.SimpleNamespace(
        __version__="9.9.9-fake",
        version=types.SimpleNamespace(cuda=cuda_build),
        cuda=cuda,
        backends=types.SimpleNamespace(mps=mps) if mps is not None else types.SimpleNamespace(),
        get_num_threads=lambda: 4,
    )
    if xpu is not None:
        torch.xpu = xpu
    return torch


def _by_name(caps):
    return {c.name: c for c in caps}


def test_cuda_build_without_gpu_reports_unavailable_with_reason(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch_directml", None)
    caps = _by_name(backend_probe.probe_backends(_fake_torch(_FakeCuda(False)), run_compute_test=False))
    cuda = caps["cuda"]
    assert cuda.available is False
    assert cuda.version == "12.6"
    assert "is_available() is False" in cuda.detail
    assert cuda.tested is False and cuda.test_passed is None


def test_cpu_only_torch_build_has_no_cuda_version(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch_directml", None)
    caps = _by_name(backend_probe.probe_backends(_fake_torch(_FakeCuda(False), cuda_build=None), run_compute_test=False))
    assert caps["cuda"].available is False
    assert caps["cuda"].version == ""
    assert "no CUDA support" in caps["cuda"].detail


def test_cuda_available_reports_devices(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch_directml", None)
    caps = _by_name(
        backend_probe.probe_backends(_fake_torch(_FakeCuda(True, 1, ["NVIDIA GeForce RTX 3060"])), run_compute_test=False)
    )
    assert caps["cuda"].available is True
    assert caps["cuda"].device_count == 1
    assert caps["cuda"].devices == ["NVIDIA GeForce RTX 3060"]


def test_unsupported_accelerators_handled(monkeypatch):
    """A torch build without an xpu attribute omits xpu; one whose
    probe raises reports it as undetermined (None), not False; and a
    crash in one probe never hides the others."""
    monkeypatch.setitem(sys.modules, "torch_directml", None)

    class BrokenXpu:
        def is_available(self):
            raise RuntimeError("driver exploded")

    caps = _by_name(backend_probe.probe_backends(_fake_torch(_FakeCuda(False), xpu=BrokenXpu()), run_compute_test=False))
    assert caps["xpu"].available is None
    assert "driver exploded" in caps["xpu"].detail
    assert caps["cuda"].available is False
    assert "cpu" in caps

    caps = _by_name(backend_probe.probe_backends(_fake_torch(_FakeCuda(False)), run_compute_test=False))
    assert "xpu" not in caps
    assert caps["directml"].available is False and "not installed" in caps["directml"].detail


def test_matmul_test_failure_is_recorded_not_raised():
    class ExplodingTorch:
        float32 = "float32"

        def no_grad(self):
            raise RuntimeError("device lost")

    result = backend_probe.run_matmul_test(ExplodingTorch(), "cuda:0")
    assert result.tested is True
    assert result.test_passed is False
    assert "device lost" in result.detail


def test_real_cpu_compute_test_passes():
    torch = pytest.importorskip("torch")
    result = backend_probe.run_matmul_test(torch, torch.device("cpu"))
    assert result.test_passed is True, result.detail
    assert result.test_gflops and result.test_gflops > 0


def test_failed_cpu_compute_test_means_cpu_backend_unavailable(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setattr(
        backend_probe,
        "probe_backends",
        lambda torch, run_compute_test=True: [
            backend_probe.BackendCapability(name="cpu", available=True, tested=True, test_passed=False, detail="broken")
        ],
    )
    caps = profiler._detect_inference_backends()
    assert caps.cpu_backend_available is False


def test_torch_absent_backends_are_undetermined(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    caps = profiler._detect_inference_backends()
    assert caps.torch_installed is False
    assert [c.name for c in caps.capabilities] == ["cpu"]
    assert caps.capabilities[0].available is None


def test_isolated_probe_matches_in_process_result():
    """The real child-process probe (what the worker uses) agrees with
    the in-process probe on every availability verdict."""
    pytest.importorskip("torch")
    isolated = profiler._detect_inference_backends_isolated(run_compute_test=True)
    in_process = profiler._detect_inference_backends(run_compute_test=False)
    assert isolated.pending is False
    assert isolated.torch_installed and isolated.torch_version == in_process.torch_version
    assert isolated.cuda_available == in_process.cuda_available
    assert {c.name: c.available for c in isolated.capabilities} == {c.name: c.available for c in in_process.capabilities}
    cpu = _by_name(isolated.capabilities)["cpu"]
    assert cpu.tested and cpu.test_passed is True and cpu.test_gflops > 0


class _FakeProbeProc:
    def __init__(self, stdout=b"", hang=False):
        self.returncode = 0
        self._stdout, self._hang, self.killed = stdout, hang, False

    def communicate(self, timeout=None):
        if self._hang and not self.killed:
            raise subprocess.TimeoutExpired(cmd="probe", timeout=timeout)
        return self._stdout, b""

    def poll(self):
        return None if (self._hang and not self.killed) else 0

    def kill(self):
        self.killed = True


def test_isolated_probe_timeout_is_undetermined_not_absent(monkeypatch):
    proc = _FakeProbeProc(hang=True)
    monkeypatch.setattr(profiler.subprocess, "Popen", lambda *a, **k: proc)
    result = profiler._detect_inference_backends_isolated(timeout=0.01)
    (cpu,) = result.capabilities
    assert cpu.available is None  # unknown -- NOT "unavailable"
    assert "could not run" in cpu.detail and "TimeoutExpired" in cpu.detail
    assert result.cuda_available is None
    assert proc.killed, "a timed-out probe child must be killed"


def test_isolated_probe_ignores_stray_output(monkeypatch):
    monkeypatch.setattr(profiler.subprocess, "Popen", lambda *a, **k: _FakeProbeProc(b"some library warning\nnot json\n"))
    assert profiler._detect_inference_backends_isolated().capabilities[0].available is None


def test_profiler_close_kills_running_probe_child():
    p = profiler.HardwareProfiler()
    p._probe_proc = proc = _FakeProbeProc(hang=True)
    p.close()
    assert proc.killed


def test_profile_registers_before_backend_probe(no_gpu_sources, monkeypatch):
    def _fail(*a, **k):
        raise AssertionError("the probe must not run when not waiting for it")

    monkeypatch.setattr(profiler, "_detect_inference_backends", _fail)
    monkeypatch.setattr(profiler, "_detect_inference_backends_isolated", _fail)
    p = profiler.HardwareProfiler(isolated_backend_probe=True)
    p._sampler = None
    snap = p.refresh_static(wait_for_backends=False)
    assert snap.backend_probe_pending is True
    assert snap.backend_capabilities == [] and snap.cuda_available is None
    assert not p.backends_ready
    assert _build_hardware_info(snap).backend_probe_pending is True

    monkeypatch.setattr(
        profiler,
        "_detect_inference_backends_isolated",
        lambda run_compute_test=True, on_start=None: profiler._BackendCapabilities(True, "x", True, False, 0, False),
    )
    p.probe_backends()
    assert p.backends_ready
    snap = p.refresh_static(reprobe_backends=False, wait_for_backends=False)
    assert snap.backend_probe_pending is False and snap.cpu_backend_available is True


def test_backend_ground_truth_computer_1():
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("CUDA available -- not Computer 1")
    caps = profiler._detect_inference_backends()
    by = _by_name(caps.capabilities)
    assert caps.cuda_available is False
    assert by["cpu"].test_passed is True
    assert by["cuda"].available is False


# ---------------------------------------------------------------------
# Profiles: source, versioning, wire format
# ---------------------------------------------------------------------


def test_collect_produces_versioned_real_profile(no_gpu_sources):
    before = int(time.time() * 1000)
    snap = profiler.collect(run_compute_test=False)
    assert snap.hardware_schema_version == profiler.HARDWARE_SCHEMA_VERSION == 3
    assert snap.profile_source == profiler.PROFILE_SOURCE_REAL
    assert snap.simulated_profile_name == ""
    assert snap.profile_collected_at_unix_ms >= before
    assert snap.cpu_physical_cores <= snap.cpu_logical_cores


def test_hand_built_snapshot_never_claims_real():
    """A Phase 1/2-shaped object built by keyword is schema 1 with an
    UNSPECIFIED source -- and goes over the wire that way."""
    snap = HardwareSnapshot(
        hostname="h", cpu_logical_cores=4, total_ram_bytes=1, has_cuda_gpu=False,
        gpu_name="", gpu_vram_bytes=0, os="x", python_version="3",
    )
    assert snap.profile_source == profiler.PROFILE_SOURCE_UNSPECIFIED
    info = _build_hardware_info(snap)
    assert info.hardware_schema_version == 1
    assert info.profile_source == common_pb2.PROFILE_SOURCE_UNSPECIFIED


def test_wire_format_preserves_unknown_vs_zero(no_gpu_sources):
    snap = profiler.collect(run_compute_test=False)
    snap.gpus = [GPUInfo("Intel", "iGPU", None, "shared", id="luid:A", shared_memory_bytes=8 * GIB, integrated=True)]
    snap.cpu_load_average_1m = None
    info = _build_hardware_info(snap)
    assert info.profile_source == common_pb2.PROFILE_SOURCE_REAL
    assert info.hardware_schema_version == 3
    assert not info.HasField("cpu_load_average_1m")
    gpu = info.gpus[0]
    assert not gpu.HasField("memory_total_bytes")
    assert not gpu.HasField("dedicated_memory_bytes")
    assert gpu.shared_memory_bytes == 8 * GIB and gpu.integrated
    assert {c.name for c in info.backend_capabilities} >= {"cpu"}


# ---------------------------------------------------------------------
# HardwareProfiler: static cached, dynamic refreshed
# ---------------------------------------------------------------------


def test_profiler_caches_static_profile(monkeypatch):
    calls = []

    def fake_collect(run_compute_test=True, backends=None):
        calls.append(1)
        return HardwareSnapshot("h", 4, 8 * GIB, False, "", 0, "x", "3", profile_source="REAL")

    monkeypatch.setattr(profiler, "collect", fake_collect)
    monkeypatch.setattr(profiler, "_detect_inference_backends", lambda run_compute_test=True: None)
    p = profiler.HardwareProfiler()
    p._sampler = None  # no GPU counters in this test
    for _ in range(5):
        p.full_snapshot()
        p.live_metrics()
    assert len(calls) == 1, "static profile must be collected once, not per call"
    p.refresh_static()
    assert len(calls) == 2


def test_periodic_refresh_reuses_backend_probe(monkeypatch, no_gpu_sources):
    """The backend probe (torch import + compute test) runs at startup
    and on reconnection, not on every periodic inventory refresh."""
    probes = []

    def fake_probe(run_compute_test=True):
        probes.append(run_compute_test)
        return profiler._BackendCapabilities(True, "x", True, False, 0, False)

    monkeypatch.setattr(profiler, "_detect_inference_backends", fake_probe)
    p = profiler.HardwareProfiler(run_compute_test=True)
    p._sampler = None
    p.refresh_static()  # startup
    p.refresh_static(reprobe_backends=False)  # periodic
    p.refresh_static(reprobe_backends=False)
    assert probes == [True]
    snap = p.refresh_static(reprobe_backends=True)  # reconnection
    assert probes == [True, True]
    assert snap.torch_installed is True and snap.cuda_available is False


def test_profiler_live_metrics_never_redetect_hardware(monkeypatch):
    p = profiler.HardwareProfiler()
    p._static = HardwareSnapshot("h", 4, 8 * GIB, False, "", 0, "x", "3")

    def _fail(*a, **k):
        raise AssertionError("live metrics must not run full detection")

    monkeypatch.setattr(profiler, "collect", _fail)
    monkeypatch.setattr(profiler, "_detect_gpus", _fail)
    monkeypatch.setattr(profiler, "_detect_inference_backends", _fail)
    for _ in range(3):
        p.live_metrics()


def test_profiler_gpu_metrics_respect_interval():
    class CountingSampler:
        supported = True

        def __init__(self):
            self.calls = 0

        def sample(self):
            self.calls += 1
            return {"luid:A": windows_gpu.GPUCounterSample(utilization_percent=1.0)}

        def close(self):
            pass

    p = profiler.HardwareProfiler(gpu_metrics_interval_seconds=3600)
    p._static = HardwareSnapshot("h", 4, 8 * GIB, False, "", 0, "x", "3", gpus=[GPUInfo("Intel", "g", None, "shared", id="luid:A")])
    p._sampler = sampler = CountingSampler()
    first = p.live_metrics()
    second = p.live_metrics()
    third = p.live_metrics()
    assert sampler.calls == 1
    assert len(first.gpu_metrics) == 1
    # Between samples: empty list = "not sampled", not "no GPU".
    assert second.gpu_metrics == [] and third.gpu_metrics == []
    p.live_metrics(force_gpu=True)
    assert sampler.calls == 2


def test_apply_live_metrics_updates_only_dynamic_fields():
    snap = HardwareSnapshot("h", 4, 8 * GIB, False, "", 0, "x", "3", cpu_model="M", ram_available_bytes=1)
    live = LiveMetrics(50.0, 2 * GIB, 6 * GIB, 75.0, swap_used_bytes=5)
    out = profiler.apply_live_metrics(snap, live)
    assert out.cpu_model == "M"
    assert out.ram_available_bytes == 2 * GIB and out.swap_used_bytes == 5
    assert snap.ram_available_bytes == 1  # original untouched
