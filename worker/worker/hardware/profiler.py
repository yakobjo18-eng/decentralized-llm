"""Hardware profiling and capability detection.

Phase 1/2 kept this deliberately small: hostname, logical core count,
total RAM, and a yes/no NVIDIA GPU check via nvidia-smi. Phase 3
expands it into the fuller picture the master prompt's hardware
profiling module calls for, while keeping every Phase 1/2 field name
and the collect() function's name/signature unchanged (the Phase 1/2
test suite and RegistryClient construct and consume HardwareSnapshot
by keyword, so additive fields with defaults are fully backward
compatible -- see docs/HARDWARE_PROFILING.md for why this shape was
chosen over a parallel v2 class).

Two different costs are deliberately kept separate (see docs/
HARDWARE_PROFILING.md, "Live vs. full"):

  - collect_live_metrics(): cheap (psutil only), meant to be called on
    every heartbeat (every few seconds).
  - collect(): the full snapshot, including GPU enumeration (shells
    out to nvidia-smi and PowerShell) and PyTorch capability checks.
    Comparatively expensive -- meant to be called at startup and only
    occasionally thereafter (worker/main.py's hardware_refresh_interval_seconds,
    default 60s), never on every heartbeat.

Nothing here installs anything. PyTorch, torch-directml, etc. are only
ever imported if already present; if the import fails, that failure
*is* the answer ("not installed"), not an error to recover from.

Nothing here fabricates a value. Any field that could not be
genuinely determined is left at an explicit "unknown" value (empty
string, 0, or None depending on the field -- see each field's
docstring in docs/HARDWARE_CAPABILITIES.md) rather than guessed.

Schema version 3 (this phase's second pass) adds: profile source
(REAL / SIMULATED) and timestamp, CPU frequency and load average, swap
(memory pressure), DXGI-based GPU memory figures with an explicit
dedicated/shared split, per-GPU live metrics from PDH counters
(collect_gpu_metrics), and per-backend capability results including a
small verified compute test (backend_probe.py). HardwareProfiler wraps
all of this with caching: static hardware is collected once and only
re-collected on request; dynamic metrics are cheap and sampled often.
"""

from __future__ import annotations

import dataclasses
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from worker.hardware import backend_probe, windows_gpu
from worker.hardware.backend_probe import BackendCapability

HARDWARE_SCHEMA_VERSION = 3

PROFILE_SOURCE_REAL = "REAL"
PROFILE_SOURCE_SIMULATED = "SIMULATED"
# A snapshot built by hand (an old fixture, a Phase 1/2-shaped object)
# never claims to be REAL -- only collect() stamps that.
PROFILE_SOURCE_UNSPECIFIED = "UNSPECIFIED"

_GIB = 1024**3

_VENDOR_ID_NAMES = {
    "GenuineIntel": "Intel",
    "AuthenticAMD": "AMD",
}

# PCI vendor IDs, as they appear in a Windows PNPDeviceID
# (e.g. "PCI\\VEN_10DE&DEV_..."). Used to identify a GPU's vendor from
# WMI data without relying on the device's display name.
_PCI_VENDOR_IDS = {
    "10DE": "NVIDIA",
    "1002": "AMD",
    "1022": "AMD",
    "8086": "Intel",
}


# ---------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------


@dataclasses.dataclass
class GPUInfo:
    vendor: str  # "NVIDIA" | "AMD" | "Intel" | "Unknown"
    name: str
    # None when genuinely unknown -- never a guessed/fabricated number.
    memory_total_bytes: Optional[int]
    memory_type: str  # "dedicated" | "shared" | "unknown"
    backends: List[str] = dataclasses.field(default_factory=list)
    driver_version: str = ""
    # --- schema 3 ---
    id: str = ""  # DXGI LUID ("luid:0x..._0x...") when known
    # Raw DXGI figures; see GPUInfo in shared/proto/common.proto for
    # exactly what each means. None = not measured.
    dedicated_memory_bytes: Optional[int] = None
    shared_memory_bytes: Optional[int] = None
    memory_source: str = ""  # "nvidia-smi" | "dxgi" | "wmi" | ""
    integrated: bool = False


@dataclasses.dataclass
class GPUMetrics:
    gpu_id: str
    utilization_percent: Optional[float] = None
    dedicated_memory_used_bytes: Optional[int] = None
    shared_memory_used_bytes: Optional[int] = None


@dataclasses.dataclass
class LiveMetrics:
    """Cheap, frequently-resampled measurements. Any field can be None
    if sampling genuinely failed (psutil missing/erroring) -- never 0
    standing in for "unknown"."""

    cpu_utilization_percent: Optional[float]
    ram_available_bytes: Optional[int]
    ram_used_bytes: Optional[int]
    ram_utilization_percent: Optional[float]
    # --- schema 3 ---
    cpu_frequency_current_mhz: Optional[float] = None
    cpu_load_average_1m: Optional[float] = None
    swap_total_bytes: Optional[int] = None
    swap_used_bytes: Optional[int] = None
    swap_utilization_percent: Optional[float] = None
    # Empty = GPU metrics not sampled this time (they have their own,
    # slower interval), NOT "no GPU".
    gpu_metrics: List[GPUMetrics] = dataclasses.field(default_factory=list)
    collected_at_unix_ms: int = 0


@dataclasses.dataclass
class HardwareSnapshot:
    # --- Phase 1/2 fields: names, types, and order UNCHANGED. ---
    hostname: str
    cpu_logical_cores: int
    total_ram_bytes: int
    has_cuda_gpu: bool
    gpu_name: str
    gpu_vram_bytes: int
    os: str
    python_version: str

    # --- Phase 3 additions, all defaulted for backward compatibility
    # with every Phase 1/2 call site that constructs HardwareSnapshot
    # by keyword without naming these. A fixture that omits
    # hardware_schema_version is, by definition, not using the Phase 3
    # shape, so defaulting it to 1 here is accurate, not just convenient. ---
    hardware_schema_version: int = 1

    cpu_manufacturer: str = "Unknown"
    cpu_model: str = "Unknown"
    cpu_physical_cores: int = 0
    cpu_architecture: str = ""

    os_name: str = ""
    os_version: str = ""
    os_build: str = ""

    gpus: List[GPUInfo] = dataclasses.field(default_factory=list)

    torch_installed: bool = False
    torch_version: Optional[str] = None
    cpu_backend_available: bool = False
    # Tri-state: None means "unknown" (torch not installed, so this
    # was never tested), not "false".
    cuda_available: Optional[bool] = None
    cuda_device_count: Optional[int] = None
    directml_available: Optional[bool] = None

    cpu_utilization_percent: Optional[float] = None
    ram_available_bytes: Optional[int] = None
    ram_used_bytes: Optional[int] = None
    ram_utilization_percent: Optional[float] = None

    # --- schema 3 ---
    profile_source: str = PROFILE_SOURCE_UNSPECIFIED
    profile_collected_at_unix_ms: int = 0
    simulated_profile_name: str = ""

    cpu_frequency_current_mhz: Optional[float] = None
    cpu_frequency_max_mhz: Optional[float] = None
    cpu_load_average_1m: Optional[float] = None

    swap_total_bytes: Optional[int] = None
    swap_used_bytes: Optional[int] = None
    swap_utilization_percent: Optional[float] = None

    backend_capabilities: List[BackendCapability] = dataclasses.field(default_factory=list)
    torch_cuda_build_version: str = ""

    gpu_metrics: List[GPUMetrics] = dataclasses.field(default_factory=list)
    # True until the (slow, separate-process) backend probe finishes;
    # while true the torch/backend fields above say nothing.
    backend_probe_pending: bool = False


def _now_ms() -> int:
    return int(time.time() * 1000)


# ---------------------------------------------------------------------
# CPU
# ---------------------------------------------------------------------


def _cpu_identity() -> Tuple[str, str]:
    """Returns (manufacturer, model). On Windows, reads the registry
    key Windows' own System Information / Task Manager read from
    (HKLM\\HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0), which
    gives the friendly marketing name (e.g. "13th Gen Intel(R) Core(TM)
    i7-1355U") -- platform.processor() alone only gives a raw CPUID
    string ("Intel64 Family 6 Model ..."). Falls back to parsing that
    raw string if the registry read fails (non-Windows, permissions,
    etc.)."""
    try:
        import winreg  # Windows-only stdlib module

        key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
            model, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            vendor_id, _ = winreg.QueryValueEx(key, "VendorIdentifier")
        manufacturer = _VENDOR_ID_NAMES.get(vendor_id, vendor_id or "Unknown")
        return manufacturer, (model or "Unknown").strip()
    except Exception:
        raw = (platform.processor() or "").strip()
        lowered = raw.lower()
        if "intel" in lowered:
            manufacturer = "Intel"
        elif "amd" in lowered:
            manufacturer = "AMD"
        else:
            manufacturer = "Unknown"
        return manufacturer, raw or "Unknown"


def _cpu_counts() -> Tuple[int, int]:
    """Returns (physical_cores, logical_cores). Deliberately two
    separate psutil calls -- conflating them would misreport a
    hybrid-core CPU (e.g. this project's Computer 1: 10 physical
    cores, 12 logical processors) as having more physical cores than
    it does."""
    try:
        import psutil

        physical = psutil.cpu_count(logical=False) or 0
        logical = psutil.cpu_count(logical=True) or 0
        return physical, logical
    except Exception:
        return 0, os.cpu_count() or 0


def _cpu_utilization_percent() -> Optional[float]:
    try:
        import psutil

        # interval=None: non-blocking, compares against the last call's
        # internal counters. The very first call in a process has
        # nothing to compare against and returns a near-meaningless
        # low/0.0 value -- expected and harmless, since this is called
        # repeatedly (every heartbeat) and later calls are meaningful.
        return float(psutil.cpu_percent(interval=None))
    except Exception:
        return None


def _cpu_frequency_mhz() -> Tuple[Optional[float], Optional[float]]:
    """Returns (current_mhz, max_mhz), each None if not reported.

    Windows limitation, documented rather than hidden: on Computer 1
    psutil reports max = 1700 MHz (the i7-1355U's base clock) and a
    current value that drops below it when the CPU is power-throttled
    (1457 MHz was observed) but never rises above it -- turbo/boost
    clocks are not visible this way. A 0 from psutil means "not
    reported" and becomes None here."""
    try:
        import psutil

        freq = psutil.cpu_freq()
        if freq is None:
            return None, None
        current = float(freq.current) if freq.current and freq.current > 0 else None
        maximum = float(freq.max) if freq.max and freq.max > 0 else None
        return current, maximum
    except Exception:
        return None, None


# psutil emulates getloadavg() on Windows with a background thread
# sampling the processor queue length; the first call starts that
# thread and returns zeros. The value only means something once it has
# had a minute to accumulate, so until then it is reported as unknown.
_LOAD_AVG_WARMUP_SECONDS = 60.0
_load_avg_started_at: Optional[float] = None


def _cpu_load_average_1m() -> Optional[float]:
    global _load_avg_started_at
    try:
        import psutil

        if not hasattr(psutil, "getloadavg"):
            return None
        one_minute = float(psutil.getloadavg()[0])
        if _load_avg_started_at is None:
            _load_avg_started_at = time.monotonic()
        if os.name == "nt" and time.monotonic() - _load_avg_started_at < _LOAD_AVG_WARMUP_SECONDS:
            return None
        return round(one_minute, 3)
    except Exception:
        return None


# ---------------------------------------------------------------------
# RAM
# ---------------------------------------------------------------------


def _ram_snapshot() -> Tuple[int, Optional[int], Optional[int], Optional[float]]:
    """Returns (total_bytes, available_bytes, used_bytes, percent)."""
    try:
        import psutil

        vm = psutil.virtual_memory()
        return int(vm.total), int(vm.available), int(vm.used), float(vm.percent)
    except Exception:
        # total_ram_bytes has always been a plain int (Phase 1/2); 0 is
        # the pre-existing "unknown" convention for that one field.
        # available/used/percent are Optional and genuinely None here.
        return 0, None, None, None


def _swap_snapshot() -> Tuple[Optional[int], Optional[int], Optional[float]]:
    """Returns (total, used, percent) for swap -- the Windows page file.
    Rising page-file use while RAM is near full is the clearest
    memory-pressure signal psutil exposes on Windows."""
    try:
        import psutil

        sw = psutil.swap_memory()
        return int(sw.total), int(sw.used), float(sw.percent)
    except Exception:
        return None, None, None


# ---------------------------------------------------------------------
# GPU detection (multi-vendor)
# ---------------------------------------------------------------------


def _detect_nvidia_gpus_via_smi(timeout: float = 5.0) -> List[GPUInfo]:
    """Authoritative for NVIDIA: nvidia-smi reports exact VRAM and
    driver version, unlike WMI's AdapterRAM (see
    _detect_gpus_via_powershell's docstring). Absence of nvidia-smi
    correctly yields [] -- never assumed."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return []
    try:
        output = (
            subprocess.check_output(
                [
                    nvidia_smi,
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                timeout=timeout,
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8", errors="ignore")
            .strip()
        )
    except Exception:
        return []

    gpus: List[GPUInfo] = []
    for line in output.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        name, mem_mib, driver = parts
        mem_bytes: Optional[int]
        try:
            mem_bytes = int(float(mem_mib)) * 1024 * 1024
        except ValueError:
            mem_bytes = None
        gpus.append(
            GPUInfo(
                vendor="NVIDIA",
                name=name,
                memory_total_bytes=mem_bytes,
                memory_type="dedicated",
                backends=["CUDA"],
                driver_version=driver,
            )
        )
    return gpus


def _vendor_from_pnp_device_id(pnp_id: str) -> str:
    match = re.search(r"VEN_([0-9A-Fa-f]{4})", pnp_id or "")
    if not match:
        return "Unknown"
    return _PCI_VENDOR_IDS.get(match.group(1).upper(), "Unknown")


def _detect_gpus_via_powershell(timeout: float = 5.0) -> list:
    """Enumerates ALL video controllers (any vendor, integrated or
    discrete) via WMI/CIM through PowerShell -- the same source
    Windows' own System Information reads from. `wmic` is deprecated/
    absent on recent Windows (confirmed during this project's Phase 0
    hardware survey), so PowerShell's Get-CimInstance is used instead.

    Returns a list of raw dicts (Name/AdapterRAM/DriverVersion/
    PNPDeviceID), or [] if PowerShell is unavailable or the query
    fails -- GPU detection is best-effort and must never be fatal to
    the overall hardware snapshot.

    Known limitation, documented rather than silently trusted:
    WMI's AdapterRAM is a 32-bit field, long known to misreport VRAM
    above roughly 4 GB, and for integrated GPUs it reports a
    conservative static aperture, not the real dynamically-shared
    allocation. This is why NVIDIA GPUs are cross-checked against the
    authoritative nvidia-smi figure instead (see _detect_gpus), and
    why integrated GPUs are tagged memory_type="shared" rather than
    "dedicated"."""
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if not powershell:
        return []
    script = (
        "Get-CimInstance Win32_VideoController | "
        "Select-Object Name,AdapterRAM,DriverVersion,PNPDeviceID | "
        "ConvertTo-Json"
    )
    try:
        output = (
            subprocess.check_output(
                [powershell, "-NoProfile", "-NonInteractive", "-Command", script],
                timeout=timeout,
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8", errors="ignore")
            .strip()
        )
    except Exception:
        return []
    if not output:
        return []
    try:
        data = json.loads(output)
    except Exception:
        return []
    if isinstance(data, dict):
        # Get-CimInstance | ConvertTo-Json gives a bare object (not a
        # list) when exactly one controller is found.
        data = [data]
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def _detect_gpus_via_dxgi() -> List[windows_gpu.DXGIAdapter]:
    """Hardware adapters as DXGI sees them, with the software
    rasterizer ("Microsoft Basic Render Driver") filtered out. [] on
    non-Windows or on any failure."""
    return [a for a in windows_gpu.enumerate_dxgi_adapters() if not a.is_software]


def _classify_memory_type(vendor: str, name: str, dedicated: Optional[int]) -> str:
    """"dedicated" | "shared" | "unknown".

    Rules (all evidence-based; see docs/HARDWARE_PROFILING.md):
      - With a DXGI figure: >= 1 GiB -> "dedicated" (real on-board
        VRAM). Below that, Intel/AMD -> "shared": the small "dedicated"
        figure of an integrated GPU (128 MiB on Iris Xe) is a firmware
        carve-out of system RAM, not VRAM.
      - Without DXGI: Intel -> "shared", except names containing "Arc"
        -> "unknown" (Arc is both a discrete card line AND, since
        Meteor Lake, an integrated GPU brand -- the name can't decide).
        NVIDIA -> "dedicated" (no NVIDIA integrated GPUs exist on
        Windows PCs). AMD -> "unknown" (WMI can't tell an APU from a
        discrete card)."""
    if dedicated is not None:
        if dedicated >= _GIB:
            return "dedicated"
        if vendor in ("AMD", "Intel"):
            return "shared"
        return "unknown"
    if vendor == "Intel":
        return "unknown" if re.search(r"\barc\b", name.lower()) else "shared"
    if vendor == "NVIDIA":
        return "dedicated"
    return "unknown"


def _vendor_from_name(name: str) -> str:
    lowered = name.lower()
    if "intel" in lowered:
        return "Intel"
    if "radeon" in lowered or "amd" in lowered:
        return "AMD"
    if "nvidia" in lowered or "geforce" in lowered or "quadro" in lowered:
        return "NVIDIA"
    return "Unknown"


def _apply_dxgi(gpu: GPUInfo, adapter: windows_gpu.DXGIAdapter) -> None:
    gpu.id = adapter.luid
    if gpu.memory_source == "nvidia-smi":
        # nvidia-smi is authoritative for NVIDIA dedicated VRAM; DXGI
        # only adds the id and the shared-memory ceiling.
        gpu.dedicated_memory_bytes = gpu.memory_total_bytes
        gpu.shared_memory_bytes = adapter.shared_system_memory or None
        return
    gpu.dedicated_memory_bytes = adapter.dedicated_video_memory
    gpu.shared_memory_bytes = adapter.shared_system_memory or None
    gpu.memory_source = "dxgi"
    gpu.memory_type = _classify_memory_type(gpu.vendor, gpu.name, adapter.dedicated_video_memory)
    gpu.integrated = gpu.memory_type == "shared"
    # memory_total_bytes keeps its schema-2 meaning ("this device's own
    # memory"): set for dedicated VRAM only. An integrated GPU has no
    # fixed pool of its own, so it stays None rather than reporting a
    # number that is really system RAM.
    gpu.memory_total_bytes = adapter.dedicated_video_memory if gpu.memory_type == "dedicated" else None


def _detect_gpus() -> List[GPUInfo]:
    nvidia_gpus = _detect_nvidia_gpus_via_smi()
    for g in nvidia_gpus:
        g.memory_source = "nvidia-smi" if g.memory_total_bytes is not None else ""
        g.dedicated_memory_bytes = g.memory_total_bytes
    nvidia_names = {g.name for g in nvidia_gpus}

    combined: List[GPUInfo] = list(nvidia_gpus)
    wmi_items = _detect_gpus_via_powershell()
    for item in wmi_items:
        name = str(item.get("Name") or "").strip()
        if not name or name in nvidia_names:
            # Already have an authoritative nvidia-smi entry for this
            # physical card -- don't add a second, less accurate one.
            continue

        vendor = _vendor_from_pnp_device_id(str(item.get("PNPDeviceID") or ""))
        if vendor == "Unknown":
            vendor = _vendor_from_name(name)

        memory_type = _classify_memory_type(vendor, name, None)
        raw_mem = item.get("AdapterRAM")
        # WMI's AdapterRAM is only kept as memory_total_bytes for a
        # dedicated card (and replaced by DXGI's 64-bit figure below
        # whenever DXGI can see the adapter). For an integrated GPU it
        # is an aperture size, not memory the GPU owns.
        memory_total_bytes = (
            raw_mem if isinstance(raw_mem, int) and raw_mem > 0 and memory_type == "dedicated" else None
        )

        combined.append(
            GPUInfo(
                vendor=vendor,
                name=name,
                memory_total_bytes=memory_total_bytes,
                memory_type=memory_type,
                backends=[],  # hardware-level CUDA/ROCm backend detection
                # for non-NVIDIA vendors isn't implemented -- there is no
                # AMD GPU available in this project to verify a ROCm/DirectML
                # check against (see docs/HARDWARE_PROFILING.md). Whether a
                # framework can USE any GPU is in backend_capabilities.
                driver_version=str(item.get("DriverVersion") or ""),
                memory_source="wmi" if memory_total_bytes is not None else "",
                integrated=memory_type == "shared",
            )
        )

    # Enrich with DXGI (accurate 64-bit memory, adapter id). Matched by
    # name -- DXGI's Description and WMI's Name are the same driver
    # string. If WMI produced nothing at all (PowerShell missing or
    # failing) DXGI alone still provides the inventory, so a WMI
    # failure is not mistaken for "no GPU".
    unmatched: List[windows_gpu.DXGIAdapter] = []
    for adapter in _detect_gpus_via_dxgi():
        match = next((g for g in combined if not g.id and g.name == adapter.description), None)
        if match is not None:
            _apply_dxgi(match, adapter)
        else:
            unmatched.append(adapter)
    if not wmi_items:
        for adapter in unmatched:
            vendor = _PCI_VENDOR_IDS.get(f"{adapter.vendor_id:04X}", _vendor_from_name(adapter.description))
            gpu = GPUInfo(vendor=vendor, name=adapter.description, memory_total_bytes=None, memory_type="unknown")
            _apply_dxgi(gpu, adapter)
            combined.append(gpu)
    return combined


# ---------------------------------------------------------------------
# Inference backend capability (separate from hardware detection)
# ---------------------------------------------------------------------


@dataclasses.dataclass
class _BackendCapabilities:
    torch_installed: bool
    torch_version: Optional[str]
    cpu_backend_available: bool
    cuda_available: Optional[bool]
    cuda_device_count: Optional[int]
    directml_available: Optional[bool]
    # --- schema 3 ---
    capabilities: List[BackendCapability] = dataclasses.field(default_factory=list)
    torch_cuda_build_version: str = ""
    # True = the probe hasn't run yet; every other field is meaningless.
    pending: bool = False


# What a profile reports before the (slow) backend probe has finished.
_PENDING_BACKENDS = _BackendCapabilities(
    torch_installed=False,
    torch_version=None,
    cpu_backend_available=False,
    cuda_available=None,
    cuda_device_count=None,
    directml_available=None,
    pending=True,
)


def _torch_absent(detail: str) -> _BackendCapabilities:
    return _BackendCapabilities(
        torch_installed=False,
        torch_version=None,
        cpu_backend_available=False,
        cuda_available=None,
        cuda_device_count=None,
        directml_available=None,
        capabilities=[BackendCapability(name="cpu", available=None, detail=detail)],
    )


def _summarize_backends(
    torch_version: Optional[str], cuda_build: str, capabilities: List[BackendCapability]
) -> _BackendCapabilities:
    """Derives the schema-2 summary flags from per-backend results
    (torch known to be installed)."""
    by_name: Dict[str, BackendCapability] = {c.name: c for c in capabilities}

    cuda = by_name.get("cuda")
    cuda_available = cuda.available if cuda else None
    cuda_device_count: Optional[int] = None
    if cuda_available is not None:
        cuda_device_count = cuda.device_count if cuda_available else 0

    directml = by_name.get("directml")
    cpu = by_name.get("cpu")
    # A CPU backend whose compute test ran and FAILED is not available,
    # whatever the import said.
    cpu_ok = cpu is not None and cpu.available is True and cpu.test_passed is not False

    return _BackendCapabilities(
        torch_installed=True,
        torch_version=torch_version or None,
        cpu_backend_available=cpu_ok,
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        directml_available=directml.available if directml else None,
        capabilities=capabilities,
        torch_cuda_build_version=cuda_build,
    )


def _detect_inference_backends_isolated(
    run_compute_test: bool = True, timeout: float = 180.0, on_start=None
) -> _BackendCapabilities:
    """Same result as _detect_inference_backends, computed in a child
    process (see backend_probe.py's subprocess section for why). If the
    child fails or times out, every backend is reported as
    undetermined (available=None) with the reason -- a failed probe is
    not evidence that torch or a backend is absent. on_start(proc) is
    called with the child's Popen so the caller can kill it on
    shutdown."""
    worker_root = str(Path(__file__).resolve().parents[2])
    env = dict(os.environ)
    env["PYTHONPATH"] = worker_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cmd = [sys.executable, "-m", "worker.hardware.backend_probe"]
    if not run_compute_test:
        cmd.append("--no-compute-test")
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, cwd=worker_root)
        if on_start is not None:
            on_start(proc)
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            raise
        lines = stdout.decode("utf-8", errors="replace").splitlines()
        payload = next((l[len(backend_probe.REPORT_PREFIX):] for l in reversed(lines) if l.startswith(backend_probe.REPORT_PREFIX)), None)
        if payload is None:
            raise RuntimeError(f"probe exited {proc.returncode} without a report: {stderr.decode(errors='replace')[-300:]}")
        report = json.loads(payload)
    except Exception as exc:
        failed = _torch_absent(f"backend probe could not run, so nothing is known: {type(exc).__name__}: {exc}"[:400])
        return failed

    if not report.get("torch_installed"):
        return _torch_absent(f"PyTorch is not installed; nothing was probed ({report.get('detail', '')})")
    capabilities = [BackendCapability(**c) for c in report.get("capabilities", [])]
    return _summarize_backends(report.get("torch_version"), report.get("torch_cuda_build_version", ""), capabilities)


def _detect_inference_backends(run_compute_test: bool = True) -> _BackendCapabilities:
    """Checks what's ACTUALLY installed and ACTUALLY working, as
    distinct from what the hardware could theoretically support. Never
    installs anything -- an ImportError here just means "not
    installed", which is a real, final answer, not a problem to fix.

    With run_compute_test, every available backend also gets the small
    verified matmul test from backend_probe.py, so "available" is backed
    by an actual computation, not just a framework flag."""
    try:
        import torch  # local import: never forced, never installed by this code
    except Exception:
        return _torch_absent("PyTorch is not installed; nothing was probed")

    return _summarize_backends(
        str(getattr(torch, "__version__", "") or ""),
        str(getattr(getattr(torch, "version", None), "cuda", None) or ""),
        backend_probe.probe_backends(torch, run_compute_test=run_compute_test),
    )


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------


def collect_live_metrics() -> LiveMetrics:
    """Cheap, meant to be called every heartbeat. Does NOT touch GPU
    detection, PowerShell, nvidia-smi, or GPU counters -- see the
    module docstring and collect_gpu_metrics()."""
    cpu_pct = _cpu_utilization_percent()
    _total, available, used, ram_pct = _ram_snapshot()
    freq_current, _freq_max = _cpu_frequency_mhz()
    swap_total, swap_used, swap_pct = _swap_snapshot()
    return LiveMetrics(
        cpu_utilization_percent=cpu_pct,
        ram_available_bytes=available,
        ram_used_bytes=used,
        ram_utilization_percent=ram_pct,
        cpu_frequency_current_mhz=freq_current,
        cpu_load_average_1m=_cpu_load_average_1m(),
        swap_total_bytes=swap_total,
        swap_used_bytes=swap_used,
        swap_utilization_percent=swap_pct,
        collected_at_unix_ms=_now_ms(),
    )


def collect_gpu_metrics(sampler: windows_gpu.GPUCounterSampler, gpus: List[GPUInfo]) -> List[GPUMetrics]:
    """Per-GPU live metrics for the adapters in `gpus` (matched by id).
    Cheap (one in-process PDH collection) but still kept on its own,
    slower interval than heartbeats. Returns [] when the counters are
    unavailable -- which says nothing about whether a GPU exists."""
    known_ids = {g.id for g in gpus if g.id}
    if not known_ids:
        return []
    out: List[GPUMetrics] = []
    for gpu_id, sample in sampler.sample().items():
        if gpu_id not in known_ids:
            continue  # e.g. the software rasterizer
        out.append(
            GPUMetrics(
                gpu_id=gpu_id,
                utilization_percent=sample.utilization_percent,
                dedicated_memory_used_bytes=sample.dedicated_memory_used_bytes,
                shared_memory_used_bytes=sample.shared_memory_used_bytes,
            )
        )
    return sorted(out, key=lambda m: m.gpu_id)


def collect(run_compute_test: bool = True, backends: Optional[_BackendCapabilities] = None) -> HardwareSnapshot:
    """The full snapshot. Comparatively expensive (GPU enumeration
    shells out to nvidia-smi and PowerShell; the backend probe imports
    torch and runs a tiny compute test) -- call at startup and only
    occasionally thereafter (see HardwareProfiler), never every
    heartbeat. Pass `backends` to reuse an earlier probe result instead
    of probing again."""
    manufacturer, model = _cpu_identity()
    physical_cores, logical_cores = _cpu_counts()
    total_ram, available_ram, used_ram, ram_pct = _ram_snapshot()
    cpu_pct = _cpu_utilization_percent()
    freq_current, freq_max = _cpu_frequency_mhz()
    swap_total, swap_used, swap_pct = _swap_snapshot()
    gpus = _detect_gpus()
    if backends is None:
        backends = _detect_inference_backends(run_compute_test=run_compute_test)

    # Backward-compatible Phase 1/2 singular-GPU fields: derived from
    # the first CUDA-capable GPU found, if any.
    cuda_gpu = next((g for g in gpus if "CUDA" in g.backends), None)
    has_cuda_gpu = cuda_gpu is not None
    gpu_name = cuda_gpu.name if cuda_gpu else ""
    gpu_vram_bytes = (cuda_gpu.memory_total_bytes or 0) if cuda_gpu else 0

    return HardwareSnapshot(
        hostname=socket.gethostname(),
        cpu_logical_cores=logical_cores,
        total_ram_bytes=total_ram,
        has_cuda_gpu=has_cuda_gpu,
        gpu_name=gpu_name,
        gpu_vram_bytes=gpu_vram_bytes,
        os=f"{platform.system()} {platform.release()}",
        python_version=platform.python_version(),
        hardware_schema_version=HARDWARE_SCHEMA_VERSION,
        cpu_manufacturer=manufacturer,
        cpu_model=model,
        cpu_physical_cores=physical_cores,
        cpu_architecture=platform.machine(),
        os_name=platform.system(),
        os_version=platform.release(),
        os_build=platform.version(),
        gpus=gpus,
        torch_installed=backends.torch_installed,
        torch_version=backends.torch_version,
        cpu_backend_available=backends.cpu_backend_available,
        cuda_available=backends.cuda_available,
        cuda_device_count=backends.cuda_device_count,
        directml_available=backends.directml_available,
        cpu_utilization_percent=cpu_pct,
        ram_available_bytes=available_ram,
        ram_used_bytes=used_ram,
        ram_utilization_percent=ram_pct,
        profile_source=PROFILE_SOURCE_REAL,
        profile_collected_at_unix_ms=_now_ms(),
        cpu_frequency_current_mhz=freq_current,
        cpu_frequency_max_mhz=freq_max,
        cpu_load_average_1m=_cpu_load_average_1m(),
        swap_total_bytes=swap_total,
        swap_used_bytes=swap_used,
        swap_utilization_percent=swap_pct,
        backend_capabilities=backends.capabilities,
        torch_cuda_build_version=backends.torch_cuda_build_version,
        backend_probe_pending=backends.pending,
    )


def apply_live_metrics(snapshot: HardwareSnapshot, live: LiveMetrics) -> HardwareSnapshot:
    """Returns a copy of snapshot with its dynamic fields replaced by
    `live` (static fields untouched)."""
    updated = dataclasses.replace(
        snapshot,
        cpu_utilization_percent=live.cpu_utilization_percent,
        ram_available_bytes=live.ram_available_bytes,
        ram_used_bytes=live.ram_used_bytes,
        ram_utilization_percent=live.ram_utilization_percent,
        cpu_frequency_current_mhz=live.cpu_frequency_current_mhz,
        cpu_load_average_1m=live.cpu_load_average_1m,
        swap_total_bytes=live.swap_total_bytes,
        swap_used_bytes=live.swap_used_bytes,
        swap_utilization_percent=live.swap_utilization_percent,
    )
    if live.gpu_metrics:
        updated.gpu_metrics = list(live.gpu_metrics)
    return updated


class HardwareProfiler:
    """The worker's hardware profiling service.

    Separates the two costs explicitly:
      - static_snapshot(): CPU identity, GPU inventory, backend
        capabilities. Collected once, cached, and only re-collected by
        refresh_static() (worker/main.py does that every
        hardware_refresh_interval_seconds, and after a reconnection).
      - live_metrics(): CPU/RAM/swap every call; GPU counters only if
        gpu_metrics_interval_seconds has elapsed since the last GPU
        sample (otherwise gpu_metrics is left empty, which the
        orchestrator treats as "keep the previous values").

    simulated_snapshot, if given, replaces all detection: the profiler
    then never touches real hardware and returns the simulated values
    (see worker/hardware/simulated.py)."""

    def __init__(
        self,
        gpu_metrics_interval_seconds: float = 15.0,
        run_compute_test: bool = True,
        simulated_snapshot: Optional[HardwareSnapshot] = None,
        simulated_live: Optional[LiveMetrics] = None,
        isolated_backend_probe: bool = False,
    ) -> None:
        # isolated_backend_probe: run the backend probe in a child
        # process (what the worker does) instead of importing torch here.
        self._isolated = isolated_backend_probe
        self._gpu_interval = gpu_metrics_interval_seconds
        self._run_compute_test = run_compute_test
        self._simulated = simulated_snapshot
        self._simulated_live = simulated_live
        self._lock = threading.Lock()
        self._static: Optional[HardwareSnapshot] = None
        self._backends: Optional[_BackendCapabilities] = None
        self._probe_proc = None  # running backend-probe child, if any
        self._sampler = windows_gpu.GPUCounterSampler() if simulated_snapshot is None else None
        self._last_gpu_sample: Optional[float] = None

    @property
    def is_simulated(self) -> bool:
        return self._simulated is not None

    @property
    def backends_ready(self) -> bool:
        return self._simulated is not None or self._backends is not None

    def probe_backends(self) -> None:
        """Runs the backend probe (torch import + compute test; 8-14 s
        on Computer 1) and caches the result. Blocking -- the worker
        calls it on a background thread."""
        if self._simulated is not None:
            return
        if self._isolated:

            def _track(proc) -> None:
                self._probe_proc = proc

            result = _detect_inference_backends_isolated(run_compute_test=self._run_compute_test, on_start=_track)
            self._probe_proc = None
        else:
            result = _detect_inference_backends(run_compute_test=self._run_compute_test)
        with self._lock:
            self._backends = result

    def refresh_static(self, reprobe_backends: bool = True, wait_for_backends: bool = True) -> HardwareSnapshot:
        """Re-collects the static profile. With reprobe_backends=False
        the cached backend probe is reused: what a framework can run on
        doesn't change while the process runs, so the worker never
        re-probes after startup. With wait_for_backends=False and no
        cached probe, the profile is returned immediately with
        backend_probe_pending=True instead of waiting for the probe."""
        if self._simulated is not None:
            snap = dataclasses.replace(self._simulated, profile_collected_at_unix_ms=_now_ms())
        else:
            if (self._backends is None or reprobe_backends) and wait_for_backends:
                self.probe_backends()
            with self._lock:
                backends = self._backends
            snap = collect(run_compute_test=self._run_compute_test, backends=backends or _PENDING_BACKENDS)
        with self._lock:
            self._static = snap
        return snap

    def static_snapshot(self) -> HardwareSnapshot:
        with self._lock:
            cached = self._static
        return cached if cached is not None else self.refresh_static()

    def live_metrics(self, force_gpu: bool = False) -> LiveMetrics:
        if self._simulated is not None:
            live = self._simulated_live or LiveMetrics(None, None, None, None)
            return dataclasses.replace(live, collected_at_unix_ms=_now_ms())
        live = collect_live_metrics()
        now = time.monotonic()
        due = self._last_gpu_sample is None or now - self._last_gpu_sample >= self._gpu_interval
        if (force_gpu or due) and self._sampler is not None and self._sampler.supported:
            self._last_gpu_sample = now
            live.gpu_metrics = collect_gpu_metrics(self._sampler, self.static_snapshot().gpus)
        return live

    def full_snapshot(self) -> HardwareSnapshot:
        """Cached static profile + freshly sampled dynamic metrics
        (including GPU counters) -- what Register() sends."""
        return apply_live_metrics(self.static_snapshot(), self.live_metrics(force_gpu=True))

    def close(self) -> None:
        if self._sampler is not None:
            self._sampler.close()
        # Don't leave a ~500 MB torch-importing child behind on a
        # graceful shutdown. (A hard-killed worker can't do this; its
        # child then exits by itself when the probe finishes.)
        proc = self._probe_proc
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass

