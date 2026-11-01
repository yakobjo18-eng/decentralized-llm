"""Phase 3 hardware detection tests: CPU-only systems, missing NVIDIA
drivers, unavailable GPU metrics, incomplete hardware information, and
the physical-vs-logical core distinction the project was explicitly
asked not to conflate.

These use monkeypatching to simulate hardware this project does not
actually have access to (NVIDIA drivers missing/erroring, PowerShell
unavailable) -- Phase 1/2's test_hardware_profiler.py already covers
the real, unmocked ground truth of this specific machine (Computer 1).
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from worker.hardware import profiler


# ---------------------------------------------------------------------
# CPU: physical vs. logical (the specific thing we were told not to
# get wrong)
# ---------------------------------------------------------------------


def test_physical_cores_never_exceed_logical_cores():
    physical, logical = profiler._cpu_counts()
    assert physical > 0
    assert logical > 0
    assert physical <= logical, (
        f"got physical={physical} > logical={logical}, which is impossible -- "
        "a core count detection bug, not a real CPU"
    )


def test_computer_1_ground_truth_not_conflated():
    """Computer 1 (the machine this test suite runs on in this
    project) is a verified 10-physical-core / 12-logical-processor
    CPU (Intel Core i7-1355U, a hybrid P-core/E-core design). This
    test only means something on that specific machine; on Computer 2
    or 3 these exact numbers will likely differ, which is fine --
    what must never happen anywhere is reporting the logical count
    (12) as the physical count."""
    physical, logical = profiler._cpu_counts()
    if logical != 12:
        pytest.skip(
            f"this machine reports {logical} logical processors, not 12 -- "
            "not Computer 1 (or its hardware changed); skipping the "
            "ground-truth assertion rather than failing on the wrong machine"
        )
    assert physical == 10, (
        f"Computer 1 has 10 physical cores; got {physical}. If this machine's 12 "
        "logical processors are being reported as 12 physical cores, that is "
        "exactly the bug this project was explicitly told to avoid."
    )
    assert physical != logical


# ---------------------------------------------------------------------
# GPU: missing drivers, erroring tools, CPU-only systems
# ---------------------------------------------------------------------


def test_missing_nvidia_smi_yields_no_nvidia_gpus(monkeypatch):
    monkeypatch.setattr(profiler.shutil, "which", lambda name: None)
    assert profiler._detect_nvidia_gpus_via_smi() == []


def test_nvidia_smi_present_but_erroring_does_not_raise(monkeypatch):
    monkeypatch.setattr(profiler.shutil, "which", lambda name: "/fake/nvidia-smi")

    def _boom(*args, **kwargs):
        raise subprocess.CalledProcessError(returncode=1, cmd=args)

    monkeypatch.setattr(profiler.subprocess, "check_output", _boom)
    assert profiler._detect_nvidia_gpus_via_smi() == []


def test_nvidia_smi_timeout_does_not_raise(monkeypatch):
    monkeypatch.setattr(profiler.shutil, "which", lambda name: "/fake/nvidia-smi")

    def _timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5)

    monkeypatch.setattr(profiler.subprocess, "check_output", _timeout)
    assert profiler._detect_nvidia_gpus_via_smi() == []


def test_powershell_unavailable_yields_empty_wmi_list(monkeypatch):
    monkeypatch.setattr(profiler.shutil, "which", lambda name: None)
    assert profiler._detect_gpus_via_powershell() == []


def test_powershell_returns_malformed_json_does_not_raise(monkeypatch):
    monkeypatch.setattr(profiler.shutil, "which", lambda name: "powershell")
    monkeypatch.setattr(
        profiler.subprocess,
        "check_output",
        lambda *a, **k: b"not valid json {{{",
    )
    assert profiler._detect_gpus_via_powershell() == []


def test_cpu_only_system_detect_gpus_returns_empty(monkeypatch):
    """Simulates a fully CPU-only system: no nvidia-smi, no WMI/PowerShell
    data at all. Must yield an empty GPU list, not a crash or a
    fabricated entry.

    Updated in Phase 3 (schema 3): DXGI is now a third GPU source and
    deliberately still enumerates GPUs when nvidia-smi and WMI both
    fail (so a PowerShell failure isn't mistaken for "no GPU"). A
    machine with NO GPU has no DXGI hardware adapters either, so it is
    stubbed too -- otherwise this test would see Computer 1's real Iris
    Xe and no longer simulate a CPU-only system. Assertion unchanged."""
    monkeypatch.setattr(profiler, "_detect_nvidia_gpus_via_smi", lambda: [])
    monkeypatch.setattr(profiler, "_detect_gpus_via_powershell", lambda: [])
    monkeypatch.setattr(profiler, "_detect_gpus_via_dxgi", lambda: [])
    assert profiler._detect_gpus() == []


def test_incomplete_hardware_information_collect_still_succeeds(monkeypatch):
    """Everything GPU-related fails/returns nothing (the "incomplete
    hardware information" case) -- collect() must still return a
    valid, complete HardwareSnapshot rather than raising.

    Phase 3 (schema 3): the DXGI source is stubbed as well, for the
    same reason as in test_cpu_only_system_detect_gpus_returns_empty."""
    monkeypatch.setattr(profiler, "_detect_nvidia_gpus_via_smi", lambda: [])
    monkeypatch.setattr(profiler, "_detect_gpus_via_powershell", lambda: [])
    monkeypatch.setattr(profiler, "_detect_gpus_via_dxgi", lambda: [])

    snapshot = profiler.collect()

    assert snapshot.gpus == []
    assert snapshot.has_cuda_gpu is False
    assert snapshot.gpu_name == ""
    assert snapshot.gpu_vram_bytes == 0
    assert snapshot.hostname
    assert snapshot.hardware_schema_version == profiler.HARDWARE_SCHEMA_VERSION


def test_vendor_from_pnp_device_id():
    assert profiler._vendor_from_pnp_device_id(r"PCI\VEN_10DE&DEV_1234") == "NVIDIA"
    assert profiler._vendor_from_pnp_device_id(r"PCI\VEN_1002&DEV_1234") == "AMD"
    assert profiler._vendor_from_pnp_device_id(r"PCI\VEN_8086&DEV_1234") == "Intel"
    assert profiler._vendor_from_pnp_device_id(r"PCI\VEN_FFFF&DEV_1234") == "Unknown"
    assert profiler._vendor_from_pnp_device_id("") == "Unknown"
    assert profiler._vendor_from_pnp_device_id("garbage") == "Unknown"


def test_detect_gpus_does_not_duplicate_nvidia_card_reported_by_both_sources(monkeypatch):
    monkeypatch.setattr(
        profiler,
        "_detect_nvidia_gpus_via_smi",
        lambda: [
            profiler.GPUInfo(
                vendor="NVIDIA",
                name="NVIDIA GeForce RTX 4090",
                memory_total_bytes=24 * 1024**3,
                memory_type="dedicated",
                backends=["CUDA"],
                driver_version="551.23",
            )
        ],
    )
    monkeypatch.setattr(
        profiler,
        "_detect_gpus_via_powershell",
        lambda: [
            # Same physical card, as WMI sees it -- must not become a
            # second entry.
            {
                "Name": "NVIDIA GeForce RTX 4090",
                "AdapterRAM": 4294967295,  # the classic WMI 32-bit overflow value
                "DriverVersion": "31.0.15.5123",
                "PNPDeviceID": r"PCI\VEN_10DE&DEV_2684",
            },
            # A genuinely separate integrated GPU.
            {
                "Name": "Intel(R) Iris(R) Xe Graphics",
                "AdapterRAM": 2147479552,
                "DriverVersion": "32.0.101.6881",
                "PNPDeviceID": r"PCI\VEN_8086&DEV_46A6",
            },
        ],
    )

    gpus = profiler._detect_gpus()

    assert len(gpus) == 2
    names = {g.name for g in gpus}
    assert names == {"NVIDIA GeForce RTX 4090", "Intel(R) Iris(R) Xe Graphics"}

    nvidia = next(g for g in gpus if g.vendor == "NVIDIA")
    assert nvidia.memory_total_bytes == 24 * 1024**3  # the accurate nvidia-smi figure, not WMI's
    assert nvidia.memory_type == "dedicated"

    intel = next(g for g in gpus if g.vendor == "Intel")
    assert intel.memory_type == "shared"  # integrated -- never "dedicated"


def test_unknown_adapter_ram_is_none_not_zero(monkeypatch):
    """WMI can report AdapterRAM as 0 or missing for some drivers.
    This must come through as None (genuinely unknown), never as a
    fabricated 0 pretending to be a measured value."""
    monkeypatch.setattr(profiler, "_detect_nvidia_gpus_via_smi", lambda: [])
    monkeypatch.setattr(
        profiler,
        "_detect_gpus_via_powershell",
        lambda: [
            {
                "Name": "Some GPU",
                "AdapterRAM": 0,
                "DriverVersion": "1.0",
                "PNPDeviceID": r"PCI\VEN_1002&DEV_0000",
            }
        ],
    )
    gpus = profiler._detect_gpus()
    assert len(gpus) == 1
    assert gpus[0].memory_total_bytes is None


# ---------------------------------------------------------------------
# Inference backend capability: tested separately from hardware
# detection, per objective 5
# ---------------------------------------------------------------------


def test_torch_not_installed_yields_unknown_not_false(monkeypatch):
    """Forces `import torch` to fail (ImportError), regardless of
    whether torch happens to be installed in whatever environment runs
    this test, so the "not installed" branch is exercised
    deterministically rather than only incidentally on a machine where
    it's really absent (as it currently is on this project's
    Computer 1)."""
    monkeypatch.setitem(sys.modules, "torch", None)

    caps = profiler._detect_inference_backends()

    assert caps.torch_installed is False
    assert caps.torch_version is None
    assert caps.cpu_backend_available is False
    assert caps.cuda_available is None  # unknown, not False
    assert caps.cuda_device_count is None
    assert caps.directml_available is None


def test_collect_reports_unknown_cuda_when_torch_absent(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    snapshot = profiler.collect()
    assert snapshot.torch_installed is False
    assert snapshot.cuda_available is None


# ---------------------------------------------------------------------
# Live metrics never touch GPU detection (the "avoid excessive
# polling" / "don't repeatedly invoke expensive GPU detection"
# requirement)
# ---------------------------------------------------------------------


def test_collect_live_metrics_never_calls_gpu_detection(monkeypatch):
    def _fail(*args, **kwargs):
        raise AssertionError("collect_live_metrics() must not touch GPU detection")

    monkeypatch.setattr(profiler, "_detect_nvidia_gpus_via_smi", _fail)
    monkeypatch.setattr(profiler, "_detect_gpus_via_powershell", _fail)
    monkeypatch.setattr(profiler, "_detect_inference_backends", _fail)

    metrics = profiler.collect_live_metrics()

    assert isinstance(metrics.cpu_utilization_percent, (float, type(None)))
    assert isinstance(metrics.ram_utilization_percent, (float, type(None)))
