"""Worker resources, in the Phase 3 "resources" shape.

Three sources, all producing the same WorkerResources:

  from_orchestrator(url)       live: GET /api/resources (+ /api/workers
                               for hostnames). REAL and SIMULATED workers
                               arrive in separate lists and keep that label.
  from_simulated_profile(path) offline: a Phase 3 YAML profile
                               (worker/profiles/simulated/), always SIMULATED.
  from_local_machine(id)       profiles THIS machine with the Phase 3
                               profiler, always REAL.

The offline paths compute "usable" memory with the same formula as the
orchestrator (orchestrator/internal/capability ComputeResources --
available minus max(10% of total, 1 GiB) for RAM; total - used -
max(10%, 0.5 GiB) for dedicated VRAM). A test checks this parity
against the live orchestrator.

Nothing here invents a value: unknown available RAM stays None, and the
planner then treats the worker as unusable (conservative), never as
having its installed RAM free.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REAL = "REAL"
SIMULATED = "SIMULATED"
UNKNOWN = "UNKNOWN"
GIB = 2**30

# Which accelerator backend drives which GPU vendor (mirrors
# capability.go's backendVendors). directml/vulkan: any vendor.
_BACKEND_VENDORS = {"cuda": ("NVIDIA",), "rocm": ("AMD",), "hip": ("AMD",), "xpu": ("Intel",), "mps": ("Apple",)}


@dataclasses.dataclass(frozen=True)
class Phase3Margins:
    """Mirrors orchestrator.yaml `capability:` defaults."""

    ram_percent: float = 10.0
    ram_min_gib: float = 1.0
    vram_percent: float = 10.0
    vram_min_gib: float = 0.5

    def ram(self, total: int) -> int:
        return max(int(total * self.ram_percent / 100), int(self.ram_min_gib * GIB))

    def vram(self, total: int) -> int:
        return max(int(total * self.vram_percent / 100), int(self.vram_min_gib * GIB))


@dataclasses.dataclass(frozen=True)
class GPUResource:
    gpu_id: str
    name: str
    vendor: str
    memory_type: str
    compute_usable: bool
    vram_total_bytes: Optional[int] = None
    vram_used_bytes: Optional[int] = None
    vram_usable_bytes: Optional[int] = None
    utilization_percent: Optional[float] = None


@dataclasses.dataclass(frozen=True)
class WorkerResources:
    worker_id: str
    profile_source: str  # REAL | SIMULATED | UNKNOWN
    online: bool
    ram_total_bytes: int
    ram_available_bytes: Optional[int]
    ram_safety_margin_bytes: int
    ram_usable_bytes: Optional[int]
    cpu_physical_cores: int
    cpu_logical_cores: int
    cpu_utilization_percent: Optional[float] = None
    cpu_frequency_max_mhz: Optional[float] = None
    execution_backends: Tuple[str, ...] = ()
    backend_gflops: Tuple[Tuple[str, float], ...] = ()
    backend_probe_pending: bool = False
    gpus: Tuple[GPUResource, ...] = ()
    hostname: str = ""
    simulated_profile_name: str = ""
    origin: str = ""  # where this record came from

    @property
    def gflops(self) -> Dict[str, float]:
        return dict(self.backend_gflops)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["backend_gflops"] = self.gflops
        return d


def from_resources_json(obj: dict, hostname: str = "", simulated_profile_name: str = "", origin: str = "") -> WorkerResources:
    """Parses one entry of the orchestrator's /api/resources response."""
    gpus = tuple(
        GPUResource(
            gpu_id=g.get("gpu_id", ""),
            name=g.get("name", ""),
            vendor=g.get("vendor", ""),
            memory_type=g.get("memory_type", ""),
            compute_usable=bool(g.get("compute_usable")),
            vram_total_bytes=g.get("vram_total_bytes"),
            vram_used_bytes=g.get("vram_used_bytes"),
            vram_usable_bytes=g.get("vram_usable_bytes"),
            utilization_percent=g.get("utilization_percent"),
        )
        for g in obj.get("gpus") or []
    )
    return WorkerResources(
        worker_id=obj["worker_id"],
        profile_source=obj.get("profile_source") or UNKNOWN,
        online=bool(obj.get("online")),
        ram_total_bytes=int(obj.get("ram_total_bytes") or 0),
        ram_available_bytes=obj.get("ram_available_bytes"),
        ram_safety_margin_bytes=int(obj.get("ram_safety_margin_bytes") or 0),
        ram_usable_bytes=obj.get("ram_usable_bytes"),
        cpu_physical_cores=int(obj.get("cpu_physical_cores") or 0),
        cpu_logical_cores=int(obj.get("cpu_logical_cores") or 0),
        cpu_utilization_percent=obj.get("cpu_utilization_percent"),
        cpu_frequency_max_mhz=obj.get("cpu_frequency_max_mhz"),
        execution_backends=tuple(obj.get("execution_backends") or ()),
        backend_gflops=tuple(sorted((obj.get("backend_gflops") or {}).items())),
        backend_probe_pending=bool(obj.get("backend_probe_pending")),
        gpus=gpus,
        hostname=hostname,
        simulated_profile_name=simulated_profile_name,
        origin=origin,
    )


def from_orchestrator(base_url: str, timeout: float = 10.0) -> List[WorkerResources]:
    def get(path: str):
        with urllib.request.urlopen(base_url.rstrip("/") + path, timeout=timeout) as resp:
            return json.loads(resp.read().decode())

    resources = get("/api/resources")
    meta = {w["id"]: w for w in get("/api/workers")}
    out = []
    for section in ("real", "simulated", "unknown_source"):
        for r in resources.get(section) or []:
            w = meta.get(r["worker_id"], {})
            out.append(from_resources_json(r, w.get("hostname", ""), w.get("simulated_profile_name", ""),
                                           origin=f"orchestrator {base_url} ({section})"))
    return sorted(out, key=lambda w: w.worker_id)


# ---------------------------------------------------------------------
# Offline sources (use the Phase 3 worker package)
# ---------------------------------------------------------------------


def _ensure_worker_importable() -> None:
    worker_root = Path(__file__).resolve().parents[2] / "worker"
    if str(worker_root) not in sys.path and worker_root.is_dir():
        sys.path.insert(0, str(worker_root))


def from_snapshot(snapshot, worker_id: str, margins: Phase3Margins = Phase3Margins(), online: bool = True,
                  origin: str = "") -> WorkerResources:
    """Same derivation as the orchestrator's ComputeResources, from a
    Phase 3 HardwareSnapshot (dynamic fields already applied)."""
    usable_backends = [
        c for c in snapshot.backend_capabilities
        if c.available is True and c.test_passed is not False
    ]
    names = [c.name for c in usable_backends]
    accelerators = sorted({n for n in names if n != "cpu"})
    backends = (["cpu"] if "cpu" in names and not snapshot.backend_probe_pending else []) + accelerators
    gflops = tuple(sorted((c.name, float(c.test_gflops)) for c in usable_backends if c.test_gflops is not None))

    metrics = {m.gpu_id: m for m in (snapshot.gpu_metrics or [])}
    gpus = []
    for g in snapshot.gpus:
        usable_compute = any(v == g.vendor for a in accelerators for v in _BACKEND_VENDORS.get(a, (g.vendor,)))
        m = metrics.get(g.id)
        total = used = usable = None
        if g.memory_type != "shared" and not g.integrated:
            total = g.dedicated_memory_bytes if g.dedicated_memory_bytes is not None else (
                g.memory_total_bytes if g.memory_type == "dedicated" else None)
            used = m.dedicated_memory_used_bytes if m else None
            if total is not None and used is not None:
                usable = max(0, total - used - margins.vram(total))
        gpus.append(GPUResource(g.id, g.name, g.vendor, g.memory_type, usable_compute, total, used, usable,
                                m.utilization_percent if m else None))

    ram_margin = margins.ram(snapshot.total_ram_bytes)
    avail = snapshot.ram_available_bytes
    return WorkerResources(
        worker_id=worker_id,
        profile_source=snapshot.profile_source if snapshot.profile_source in (REAL, SIMULATED) else UNKNOWN,
        online=online,
        ram_total_bytes=snapshot.total_ram_bytes,
        ram_available_bytes=avail,
        ram_safety_margin_bytes=ram_margin,
        ram_usable_bytes=max(0, avail - ram_margin) if avail is not None else None,
        cpu_physical_cores=snapshot.cpu_physical_cores,
        cpu_logical_cores=snapshot.cpu_logical_cores,
        cpu_utilization_percent=snapshot.cpu_utilization_percent,
        cpu_frequency_max_mhz=snapshot.cpu_frequency_max_mhz,
        execution_backends=tuple(backends),
        backend_gflops=gflops,
        backend_probe_pending=snapshot.backend_probe_pending,
        gpus=tuple(gpus),
        hostname=snapshot.hostname,
        simulated_profile_name=snapshot.simulated_profile_name,
        origin=origin,
    )


def from_simulated_profile(path: str, worker_id: Optional[str] = None, margins: Phase3Margins = Phase3Margins()) -> WorkerResources:
    _ensure_worker_importable()
    from worker.hardware.profiler import apply_live_metrics
    from worker.hardware.simulated import load_profile

    prof = load_profile(path)
    snap = apply_live_metrics(prof.snapshot, prof.live)
    if snap.profile_source != SIMULATED:  # load_profile guarantees this; belt and braces
        raise ValueError(f"{path} did not produce a SIMULATED profile")
    return from_snapshot(snap, worker_id or prof.name, margins, origin=f"simulated profile {path}")


def from_local_machine(worker_id: str, margins: Phase3Margins = Phase3Margins(), run_compute_test: bool = True) -> WorkerResources:
    """Profiles this machine for real (Phase 3 profiler; the backend
    probe runs in a child process, ~10 s on Computer 1)."""
    _ensure_worker_importable()
    from worker.hardware.profiler import HardwareProfiler

    profiler = HardwareProfiler(isolated_backend_probe=True, run_compute_test=run_compute_test)
    try:
        profiler.refresh_static()
        snap = profiler.full_snapshot()
    finally:
        profiler.close()
    if snap.profile_source != REAL:
        raise RuntimeError("local profiling did not produce a REAL profile")
    return from_snapshot(snap, worker_id, margins, origin="local machine (Phase 3 profiler)")
