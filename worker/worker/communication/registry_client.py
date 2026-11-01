"""gRPC client for the orchestrator's WorkerRegistry service
(shared/proto/registry.proto). This is the worker's half of the Phase 1
"orchestrator and worker can communicate" requirement -- every call
here is a real network round trip over gRPC, not a stub.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional

import grpc

from worker.generated import common_pb2, registry_pb2, registry_pb2_grpc
from worker.hardware.profiler import (
    PROFILE_SOURCE_REAL,
    PROFILE_SOURCE_SIMULATED,
    GPUMetrics,
    HardwareSnapshot,
    LiveMetrics,
)

logger = logging.getLogger(__name__)

_PROFILE_SOURCES = {
    PROFILE_SOURCE_REAL: common_pb2.PROFILE_SOURCE_REAL,
    PROFILE_SOURCE_SIMULATED: common_pb2.PROFILE_SOURCE_SIMULATED,
}


def _set_optional(msg, field: str, value) -> None:
    """Sets a proto3 `optional` field only when the value is genuinely
    known, so the receiver's HasField() distinguishes "unknown" from a
    real 0/False."""
    if value is not None:
        setattr(msg, field, value)


def _build_gpu_metrics(metrics: List[GPUMetrics]) -> List[common_pb2.GPUMetrics]:
    out = []
    for m in metrics or []:
        pb = common_pb2.GPUMetrics(gpu_id=m.gpu_id)
        _set_optional(pb, "utilization_percent", m.utilization_percent)
        _set_optional(pb, "dedicated_memory_used_bytes", m.dedicated_memory_used_bytes)
        _set_optional(pb, "shared_memory_used_bytes", m.shared_memory_used_bytes)
        out.append(pb)
    return out


def _build_hardware_info(hardware: HardwareSnapshot) -> common_pb2.HardwareInfo:
    """Builds the full HardwareInfo proto from a HardwareSnapshot,
    including Phase 3 fields. Uses getattr with the dataclass's own
    defaults as a fallback so a Phase 1/2-shaped HardwareSnapshot (one
    that never got the Phase 3 fields set, e.g. an older in-memory
    object) still builds a valid message -- schema_version 1, with the
    new fields at their honest "unknown" defaults, exactly describing
    what that object actually represents."""
    gpus = []
    for g in hardware.gpus or []:
        gpu = common_pb2.GPUInfo(
            vendor=g.vendor,
            name=g.name,
            memory_total_bytes=g.memory_total_bytes,
            memory_type=g.memory_type,
            backends=list(g.backends),
            driver_version=g.driver_version,
            id=getattr(g, "id", ""),
            memory_source=getattr(g, "memory_source", ""),
            integrated=getattr(g, "integrated", False),
        )
        _set_optional(gpu, "dedicated_memory_bytes", getattr(g, "dedicated_memory_bytes", None))
        _set_optional(gpu, "shared_memory_bytes", getattr(g, "shared_memory_bytes", None))
        gpus.append(gpu)

    capabilities = []
    for c in getattr(hardware, "backend_capabilities", None) or []:
        cap = common_pb2.BackendCapability(
            name=c.name,
            device_count=c.device_count,
            version=c.version,
            devices=list(c.devices),
            tested=c.tested,
            detail=c.detail,
        )
        _set_optional(cap, "available", c.available)
        _set_optional(cap, "test_passed", c.test_passed)
        _set_optional(cap, "test_gflops", c.test_gflops)
        capabilities.append(cap)

    info = common_pb2.HardwareInfo(
        hostname=hardware.hostname,
        cpu_logical_cores=hardware.cpu_logical_cores,
        total_ram_bytes=hardware.total_ram_bytes,
        has_cuda_gpu=hardware.has_cuda_gpu,
        gpu_name=hardware.gpu_name,
        gpu_vram_bytes=hardware.gpu_vram_bytes,
        os=hardware.os,
        python_version=hardware.python_version,
        hardware_schema_version=hardware.hardware_schema_version,
        cpu_manufacturer=hardware.cpu_manufacturer,
        cpu_model=hardware.cpu_model,
        cpu_physical_cores=hardware.cpu_physical_cores,
        cpu_architecture=hardware.cpu_architecture,
        os_name=hardware.os_name,
        os_version=hardware.os_version,
        os_build=hardware.os_build,
        gpus=gpus,
        torch_installed=hardware.torch_installed,
        torch_version=hardware.torch_version or "",
        cpu_backend_available=hardware.cpu_backend_available,
        profile_source=_PROFILE_SOURCES.get(
            getattr(hardware, "profile_source", ""), common_pb2.PROFILE_SOURCE_UNSPECIFIED
        ),
        profile_collected_at_unix_ms=getattr(hardware, "profile_collected_at_unix_ms", 0),
        simulated_profile_name=getattr(hardware, "simulated_profile_name", ""),
        backend_capabilities=capabilities,
        torch_cuda_build_version=getattr(hardware, "torch_cuda_build_version", ""),
        gpu_metrics=_build_gpu_metrics(getattr(hardware, "gpu_metrics", None)),
        backend_probe_pending=getattr(hardware, "backend_probe_pending", False),
    )
    for field in (
        "cpu_frequency_current_mhz",
        "cpu_frequency_max_mhz",
        "cpu_load_average_1m",
        "swap_total_bytes",
        "swap_used_bytes",
        "swap_utilization_percent",
    ):
        _set_optional(info, field, getattr(hardware, field, None))
    # optional (tri-state) fields: only set on the proto message if
    # genuinely known, so HasField(...) on the receiving end correctly
    # reports "not set" rather than a fabricated false/0.
    if hardware.cuda_available is not None:
        info.cuda_available = hardware.cuda_available
    if hardware.cuda_device_count is not None:
        info.cuda_device_count = hardware.cuda_device_count
    if hardware.directml_available is not None:
        info.directml_available = hardware.directml_available
    if hardware.cpu_utilization_percent is not None:
        info.cpu_utilization_percent = hardware.cpu_utilization_percent
    if hardware.ram_available_bytes is not None:
        info.ram_available_bytes = hardware.ram_available_bytes
    if hardware.ram_used_bytes is not None:
        info.ram_used_bytes = hardware.ram_used_bytes
    if hardware.ram_utilization_percent is not None:
        info.ram_utilization_percent = hardware.ram_utilization_percent
    return info


class RegistryClient:
    def __init__(self, orchestrator_address: str):
        self._address = orchestrator_address
        self._channel = grpc.insecure_channel(orchestrator_address)
        self._stub = registry_pb2_grpc.WorkerRegistryStub(self._channel)

    def register(
        self,
        worker_id: str,
        own_address: str,
        hardware: HardwareSnapshot,
        timeout: float = 5.0,
    ) -> registry_pb2.RegisterResponse:
        request = registry_pb2.RegisterRequest(
            worker_id=worker_id,
            address=own_address,
            hardware=_build_hardware_info(hardware),
        )
        response = self._stub.Register(request, timeout=timeout)
        logger.info(
            "Register(%s) -> accepted=%s message=%s",
            worker_id,
            response.accepted,
            response.message,
        )
        return response

    def heartbeat(
        self,
        worker_id: str,
        status: str = "idle",
        timeout: float = 5.0,
        live_metrics: Optional[LiveMetrics] = None,
    ) -> registry_pb2.HeartbeatResponse:
        request = registry_pb2.HeartbeatRequest(
            worker_id=worker_id,
            client_time_unix_ms=int(time.time() * 1000),
            status=status,
        )
        if live_metrics is not None:
            if live_metrics.cpu_utilization_percent is not None:
                request.cpu_utilization_percent = live_metrics.cpu_utilization_percent
            if live_metrics.ram_available_bytes is not None:
                request.ram_available_bytes = live_metrics.ram_available_bytes
            if live_metrics.ram_used_bytes is not None:
                request.ram_used_bytes = live_metrics.ram_used_bytes
            if live_metrics.ram_utilization_percent is not None:
                request.ram_utilization_percent = live_metrics.ram_utilization_percent
            for field in (
                "cpu_frequency_current_mhz",
                "cpu_load_average_1m",
                "swap_total_bytes",
                "swap_used_bytes",
                "swap_utilization_percent",
            ):
                _set_optional(request, field, getattr(live_metrics, field, None))
            request.gpu_metrics.extend(_build_gpu_metrics(getattr(live_metrics, "gpu_metrics", None)))
            request.metrics_collected_at_unix_ms = getattr(live_metrics, "collected_at_unix_ms", 0)
        return self._stub.Heartbeat(request, timeout=timeout)

    def deregister(
        self, worker_id: str, reason: str = "graceful shutdown", timeout: float = 5.0
    ) -> registry_pb2.DeregisterResponse:
        request = registry_pb2.DeregisterRequest(worker_id=worker_id, reason=reason)
        response = self._stub.Deregister(request, timeout=timeout)
        logger.info("Deregister(%s) -> accepted=%s", worker_id, response.accepted)
        return response

    def close(self) -> None:
        self._channel.close()
