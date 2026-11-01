"""Simulated hardware profiles (YAML), for development and algorithm
testing while Computers 2 and 3 are physically unavailable.

A simulated profile is a hand-written description of a HYPOTHETICAL
machine. Nothing in it was measured. Every snapshot built from one is
stamped profile_source="SIMULATED" and carries SIMULATED_NOTICE, and
the orchestrator rejects simulated registrations unless it was
explicitly started with allow_simulated_workers enabled. The example
profiles shipped in worker/profiles/simulated/ are NOT the
specifications of Computer 2 or Computer 3 -- those are unknown.

Usage:
    python -m worker.hardware.simulated validate profiles/simulated/*.yaml
    python -m worker.hardware.simulated show profiles/simulated/profile_b_gpu_worker.yaml

Format: see docs/SIMULATED_HARDWARE.md. Validation is strict on
purpose (unknown keys are errors, so a typo can't silently produce a
profile with a default value the author didn't intend).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from worker.hardware.backend_probe import BackendCapability
from worker.hardware.profiler import (
    HARDWARE_SCHEMA_VERSION,
    PROFILE_SOURCE_SIMULATED,
    GPUInfo,
    GPUMetrics,
    HardwareSnapshot,
    LiveMetrics,
)

SIMULATED_NOTICE = "SIMULATED - NOT PHYSICALLY VERIFIED"
PROFILE_FORMAT_VERSION = 1

_GIB = 1024**3
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")
_VENDORS = {"NVIDIA", "AMD", "Intel", "Unknown"}
_MEMORY_TYPES = {"dedicated", "shared"}
# Which simulated compute backend a vendor's GPU could plausibly expose.
# Prevents e.g. "an Intel GPU with CUDA", which no real driver provides.
_BACKEND_VENDORS = {
    "cuda": {"NVIDIA"},
    "rocm": {"AMD"},
    "xpu": {"Intel"},
    "directml": {"NVIDIA", "AMD", "Intel"},
    "vulkan": {"NVIDIA", "AMD", "Intel"},
}

_TOP_KEYS = {
    "profile_format_version", "profile_source", "name", "description", "hostname",
    "os", "cpu", "ram", "gpus", "backends", "dynamic",
}
_OS_KEYS = {"name", "version", "build"}
_CPU_KEYS = {"manufacturer", "model", "architecture", "physical_cores", "logical_cores", "max_frequency_mhz"}
_RAM_KEYS = {"total_gib", "swap_total_gib"}
_GPU_KEYS = {"vendor", "model", "memory_type", "vram_gib", "shared_gib", "compute_backends", "driver_version"}
_BACKEND_KEYS = {"torch_installed", "torch_version", "cpu"}
_DYNAMIC_KEYS = {"cpu_utilization_percent", "ram_available_gib", "swap_used_gib", "gpus"}
_DYNAMIC_GPU_KEYS = {"utilization_percent", "vram_used_gib", "shared_used_gib"}


class SimulatedProfileError(ValueError):
    """A profile failed validation. The message lists every problem
    found, not just the first."""


@dataclasses.dataclass
class SimulatedProfile:
    name: str
    description: str
    snapshot: HardwareSnapshot
    live: LiveMetrics
    source_path: str = ""
    notice: str = SIMULATED_NOTICE


class _Errors:
    def __init__(self, source: str) -> None:
        self.source = source
        self.items: List[str] = []

    def add(self, msg: str) -> None:
        self.items.append(msg)

    def raise_if_any(self) -> None:
        if self.items:
            joined = "\n  - ".join(self.items)
            raise SimulatedProfileError(f"invalid simulated profile {self.source}:\n  - {joined}")


def _check_keys(section: Any, allowed: set, where: str, errors: _Errors) -> Dict[str, Any]:
    if section is None:
        return {}
    if not isinstance(section, dict):
        errors.add(f"{where} must be a mapping")
        return {}
    for key in section:
        if key not in allowed:
            errors.add(f"{where}: unknown key {key!r} (allowed: {', '.join(sorted(allowed))})")
    return section


def _number(section: Dict[str, Any], key: str, where: str, errors: _Errors, *, required: bool,
            minimum: float = 0.0, maximum: Optional[float] = None, integer: bool = False,
            exclusive_min: bool = False) -> Optional[float]:
    if key not in section or section[key] is None:
        if required:
            errors.add(f"{where}.{key} is required")
        return None
    value = section[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.add(f"{where}.{key} must be a number, got {value!r}")
        return None
    if integer and not float(value).is_integer():
        errors.add(f"{where}.{key} must be a whole number, got {value!r}")
        return None
    if value < minimum or (exclusive_min and value == minimum):
        errors.add(f"{where}.{key} must be {'>' if exclusive_min else '>='} {minimum}, got {value!r}")
        return None
    if maximum is not None and value > maximum:
        errors.add(f"{where}.{key} must be <= {maximum}, got {value!r}")
        return None
    return value


def _string(section: Dict[str, Any], key: str, where: str, errors: _Errors, *, required: bool,
            default: str = "") -> str:
    value = section.get(key)
    if value is None or value == "":
        if required:
            errors.add(f"{where}.{key} is required")
        return default
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        errors.add(f"{where}.{key} must be a string")
        return default
    return str(value)


def parse_profile(data: Any, source: str = "<dict>") -> SimulatedProfile:
    """Validates a parsed YAML document and builds the snapshot/live
    metrics it describes. Raises SimulatedProfileError listing every
    problem."""
    errors = _Errors(source)
    if not isinstance(data, dict):
        errors.add("top level must be a mapping")
        errors.raise_if_any()
    _check_keys(data, _TOP_KEYS, "profile", errors)

    fmt = data.get("profile_format_version")
    if fmt != PROFILE_FORMAT_VERSION:
        errors.add(f"profile_format_version must be {PROFILE_FORMAT_VERSION}, got {fmt!r}")
    if data.get("profile_source") != PROFILE_SOURCE_SIMULATED:
        # A profile file can never claim to be a real measurement.
        errors.add(f"profile_source must be exactly {PROFILE_SOURCE_SIMULATED!r}, got {data.get('profile_source')!r}")

    name = _string(data, "name", "profile", errors, required=True)
    if name and not _NAME_RE.match(name):
        errors.add(f"profile.name {name!r} must be lowercase letters, digits and dashes")
    description = _string(data, "description", "profile", errors, required=False)
    hostname = _string(data, "hostname", "profile", errors, required=False, default=f"sim-{name}")

    os_section = _check_keys(data.get("os"), _OS_KEYS, "os", errors)

    cpu = _check_keys(data.get("cpu"), _CPU_KEYS, "cpu", errors)
    if "cpu" not in data:
        errors.add("cpu section is required")
    physical = _number(cpu, "physical_cores", "cpu", errors, required=True, minimum=1, maximum=4096, integer=True)
    logical = _number(cpu, "logical_cores", "cpu", errors, required=True, minimum=1, maximum=8192, integer=True)
    if physical is not None and logical is not None and physical > logical:
        errors.add(f"cpu.physical_cores ({int(physical)}) cannot exceed cpu.logical_cores ({int(logical)})")
    max_freq = _number(cpu, "max_frequency_mhz", "cpu", errors, required=False, minimum=0, exclusive_min=True)
    cpu_model = _string(cpu, "model", "cpu", errors, required=True)

    ram = _check_keys(data.get("ram"), _RAM_KEYS, "ram", errors)
    if "ram" not in data:
        errors.add("ram section is required")
    total_gib = _number(ram, "total_gib", "ram", errors, required=True, minimum=0, exclusive_min=True, maximum=65536)
    swap_total_gib = _number(ram, "swap_total_gib", "ram", errors, required=False, minimum=0)

    gpus_raw = data.get("gpus") or []
    if not isinstance(gpus_raw, list):
        errors.add("gpus must be a list")
        gpus_raw = []
    gpus: List[GPUInfo] = []
    gpu_backends: Dict[str, List[str]] = {}
    for i, raw in enumerate(gpus_raw):
        where = f"gpus[{i}]"
        g = _check_keys(raw, _GPU_KEYS, where, errors)
        vendor = _string(g, "vendor", where, errors, required=True)
        if vendor and vendor not in _VENDORS:
            errors.add(f"{where}.vendor must be one of {sorted(_VENDORS)}, got {vendor!r}")
        model = _string(g, "model", where, errors, required=True)
        mem_type = _string(g, "memory_type", where, errors, required=True)
        if mem_type and mem_type not in _MEMORY_TYPES:
            errors.add(f"{where}.memory_type must be 'dedicated' or 'shared', got {mem_type!r}")
        vram = _number(g, "vram_gib", where, errors, required=mem_type == "dedicated", minimum=0, exclusive_min=True)
        shared = _number(g, "shared_gib", where, errors, required=False, minimum=0)
        if mem_type == "shared" and vram is not None:
            # The rule this project was given: shared system memory is
            # never reported as dedicated VRAM.
            errors.add(f"{where}: a 'shared' (integrated) GPU cannot have vram_gib; use shared_gib")
        if shared is not None and total_gib is not None and shared > total_gib:
            errors.add(f"{where}.shared_gib ({shared}) cannot exceed ram.total_gib ({total_gib})")
        backends = g.get("compute_backends") or []
        if not isinstance(backends, list) or not all(isinstance(b, str) for b in backends):
            errors.add(f"{where}.compute_backends must be a list of strings")
            backends = []
        for b in backends:
            allowed_vendors = _BACKEND_VENDORS.get(b)
            if allowed_vendors is None:
                errors.add(f"{where}.compute_backends: unknown backend {b!r} (known: {sorted(_BACKEND_VENDORS)})")
            elif vendor and vendor not in allowed_vendors:
                errors.add(f"{where}.compute_backends: {b!r} is not available on {vendor} GPUs")
        gpu_id = f"sim:{name}:{i}"
        for b in backends:
            gpu_backends.setdefault(b, []).append(model)
        gpus.append(
            GPUInfo(
                vendor=vendor or "Unknown",
                name=model,
                memory_total_bytes=int(vram * _GIB) if vram is not None and mem_type == "dedicated" else None,
                memory_type=mem_type or "unknown",
                backends=[b.upper() for b in backends],
                driver_version=_string(g, "driver_version", where, errors, required=False, default="simulated"),
                id=gpu_id,
                dedicated_memory_bytes=int(vram * _GIB) if vram is not None else None,
                shared_memory_bytes=int(shared * _GIB) if shared is not None else None,
                memory_source="simulated",
                integrated=mem_type == "shared",
            )
        )

    backends_section = _check_keys(data.get("backends"), _BACKEND_KEYS, "backends", errors)
    torch_installed = backends_section.get("torch_installed", True)
    cpu_backend = backends_section.get("cpu", True)
    for key, value in (("torch_installed", torch_installed), ("cpu", cpu_backend)):
        if not isinstance(value, bool):
            errors.add(f"backends.{key} must be true or false")
    if gpu_backends and torch_installed is False:
        errors.add("gpus declare compute_backends but backends.torch_installed is false")

    dynamic = _check_keys(data.get("dynamic"), _DYNAMIC_KEYS, "dynamic", errors)
    cpu_util = _number(dynamic, "cpu_utilization_percent", "dynamic", errors, required=False, maximum=100)
    ram_avail_gib = _number(dynamic, "ram_available_gib", "dynamic", errors, required=False)
    if ram_avail_gib is not None and total_gib is not None and ram_avail_gib > total_gib:
        errors.add(f"dynamic.ram_available_gib ({ram_avail_gib}) cannot exceed ram.total_gib ({total_gib})")
    swap_used_gib = _number(dynamic, "swap_used_gib", "dynamic", errors, required=False)
    if swap_used_gib is not None and swap_total_gib is not None and swap_used_gib > swap_total_gib:
        errors.add("dynamic.swap_used_gib cannot exceed ram.swap_total_gib")

    gpu_dyn_raw = dynamic.get("gpus") or []
    if not isinstance(gpu_dyn_raw, list):
        errors.add("dynamic.gpus must be a list")
        gpu_dyn_raw = []
    if len(gpu_dyn_raw) > len(gpus):
        errors.add(f"dynamic.gpus has {len(gpu_dyn_raw)} entries but only {len(gpus)} gpus are declared")
    gpu_metrics: List[GPUMetrics] = []
    for i, raw in enumerate(gpu_dyn_raw[: len(gpus)]):
        where = f"dynamic.gpus[{i}]"
        d = _check_keys(raw, _DYNAMIC_GPU_KEYS, where, errors)
        util = _number(d, "utilization_percent", where, errors, required=False, maximum=100)
        vram_used = _number(d, "vram_used_gib", where, errors, required=False)
        shared_used = _number(d, "shared_used_gib", where, errors, required=False)
        gpu = gpus[i]
        if vram_used is not None and (gpu.dedicated_memory_bytes is None or vram_used * _GIB > gpu.dedicated_memory_bytes):
            errors.add(f"{where}.vram_used_gib exceeds gpus[{i}]'s vram_gib")
        if shared_used is not None and total_gib is not None and shared_used > total_gib:
            errors.add(f"{where}.shared_used_gib exceeds ram.total_gib")
        gpu_metrics.append(
            GPUMetrics(
                gpu_id=gpu.id,
                utilization_percent=util,
                dedicated_memory_used_bytes=int(vram_used * _GIB) if vram_used is not None else None,
                shared_memory_used_bytes=int(shared_used * _GIB) if shared_used is not None else None,
            )
        )

    errors.raise_if_any()

    total_bytes = int(total_gib * _GIB)
    available_bytes = int(ram_avail_gib * _GIB) if ram_avail_gib is not None else None
    used_bytes = total_bytes - available_bytes if available_bytes is not None else None
    ram_pct = round(100.0 * used_bytes / total_bytes, 1) if used_bytes is not None else None
    swap_total = int(swap_total_gib * _GIB) if swap_total_gib is not None else None
    swap_used = int(swap_used_gib * _GIB) if swap_used_gib is not None else None
    swap_pct = round(100.0 * swap_used / swap_total, 1) if swap_used is not None and swap_total else None

    sim_detail = f"{SIMULATED_NOTICE}: declared by profile, never executed"
    capabilities: List[BackendCapability] = []
    if torch_installed:
        capabilities.append(BackendCapability(name="cpu", available=bool(cpu_backend), device_count=1, detail=sim_detail))
        for backend, devices in sorted(gpu_backends.items()):
            capabilities.append(
                BackendCapability(name=backend, available=True, device_count=len(devices), devices=devices, detail=sim_detail)
            )
    else:
        capabilities.append(BackendCapability(name="cpu", available=None, detail=f"{SIMULATED_NOTICE}: torch not installed"))

    cuda_gpus = [g for g in gpus if "CUDA" in g.backends]
    live = LiveMetrics(
        cpu_utilization_percent=cpu_util,
        ram_available_bytes=available_bytes,
        ram_used_bytes=used_bytes,
        ram_utilization_percent=ram_pct,
        cpu_frequency_current_mhz=max_freq,
        swap_total_bytes=swap_total,
        swap_used_bytes=swap_used,
        swap_utilization_percent=swap_pct,
        gpu_metrics=gpu_metrics,
    )
    snapshot = HardwareSnapshot(
        hostname=hostname,
        cpu_logical_cores=int(logical),
        total_ram_bytes=total_bytes,
        has_cuda_gpu=bool(cuda_gpus),
        gpu_name=cuda_gpus[0].name if cuda_gpus else "",
        gpu_vram_bytes=(cuda_gpus[0].memory_total_bytes or 0) if cuda_gpus else 0,
        os=f"{os_section.get('name', 'Simulated')} {os_section.get('version', '')}".strip(),
        python_version="",
        hardware_schema_version=HARDWARE_SCHEMA_VERSION,
        cpu_manufacturer=_string(cpu, "manufacturer", "cpu", errors, required=False, default="Unknown"),
        cpu_model=cpu_model,
        cpu_physical_cores=int(physical),
        cpu_architecture=_string(cpu, "architecture", "cpu", errors, required=False, default="AMD64"),
        os_name=str(os_section.get("name", "Simulated")),
        os_version=str(os_section.get("version", "")),
        os_build=str(os_section.get("build", "simulated")),
        gpus=gpus,
        torch_installed=bool(torch_installed),
        torch_version=str(backends_section.get("torch_version", "simulated")) if torch_installed else None,
        cpu_backend_available=bool(torch_installed and cpu_backend),
        cuda_available=bool(cuda_gpus) if torch_installed else None,
        cuda_device_count=len(cuda_gpus) if torch_installed else None,
        directml_available=("directml" in gpu_backends) if torch_installed else None,
        cpu_utilization_percent=cpu_util,
        ram_available_bytes=available_bytes,
        ram_used_bytes=used_bytes,
        ram_utilization_percent=ram_pct,
        profile_source=PROFILE_SOURCE_SIMULATED,
        simulated_profile_name=name,
        cpu_frequency_current_mhz=max_freq,
        cpu_frequency_max_mhz=max_freq,
        swap_total_bytes=swap_total,
        swap_used_bytes=swap_used,
        swap_utilization_percent=swap_pct,
        backend_capabilities=capabilities,
        gpu_metrics=gpu_metrics,
    )
    return SimulatedProfile(name=name, description=description, snapshot=snapshot, live=live, source_path=source)


def load_profile(path: str) -> SimulatedProfile:
    p = Path(path)
    try:
        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        raise SimulatedProfileError(f"simulated profile not found: {path}") from None
    except yaml.YAMLError as exc:
        raise SimulatedProfileError(f"simulated profile {path} is not valid YAML: {exc}") from None
    return parse_profile(data, source=str(p))


def load_profiles_dir(directory: str) -> List[SimulatedProfile]:
    """Loads every *.yaml in directory; raises on the first invalid one
    (a test fixture set with a broken member should fail loudly)."""
    profiles = [load_profile(str(p)) for p in sorted(Path(directory).glob("*.yaml"))]
    names = [p.name for p in profiles]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise SimulatedProfileError(f"duplicate simulated profile names in {directory}: {sorted(duplicates)}")
    return profiles


def _main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate or inspect simulated hardware profiles")
    sub = parser.add_subparsers(dest="command", required=True)
    v = sub.add_parser("validate", help="validate one or more profile files")
    v.add_argument("paths", nargs="+")
    s = sub.add_parser("show", help="print the snapshot a profile produces")
    s.add_argument("path")
    args = parser.parse_args(argv)

    if args.command == "validate":
        failed = 0
        for path in args.paths:
            try:
                prof = load_profile(path)
                print(f"OK    {path}  ({prof.name}) -- {SIMULATED_NOTICE}")
            except SimulatedProfileError as exc:
                failed += 1
                print(f"FAIL  {exc}")
        return 1 if failed else 0

    prof = load_profile(args.path)
    print(json.dumps({"notice": SIMULATED_NOTICE, **dataclasses.asdict(prof.snapshot)}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
