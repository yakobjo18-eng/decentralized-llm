"""Loads worker configuration from a YAML file with environment-variable
overrides, mirroring the pattern used by the Go orchestrator's config
package (orchestrator/internal/config) so both sides behave consistently.

Per Rule 11 (all network addresses/ports configurable, nothing
hardcoded): every setting here has a config-file key and an environment
variable override, and there is no machine-specific IP baked in
anywhere -- the default orchestrator_address is a loopback placeholder
for local development only.
"""

from __future__ import annotations

import dataclasses
import os
import socket
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv


@dataclasses.dataclass
class WorkerConfig:
    worker_id: str
    orchestrator_address: str = "127.0.0.1:9090"
    # Health server bind address. Loopback by default (Phase 7): set this
    # machine's private LAN IP explicitly to serve other machines. Public
    # IPs and hostnames are refused at startup (config/bind_policy.py);
    # 0.0.0.0 is allowed with a warning.
    listen_host: str = "127.0.0.1"
    listen_port: int = 50100
    heartbeat_interval_seconds: float = 5.0
    # Phase 3: how often the worker re-collects and re-sends its FULL
    # hardware snapshot (including GPU enumeration, comparatively
    # expensive -- see worker/hardware/profiler.py). Deliberately much
    # less frequent than heartbeat_interval_seconds, which only carries
    # cheap live CPU/RAM metrics. Default raised from 60s to 300s in
    # schema 3: each refresh starts powershell.exe, and a measured 60s
    # cadence made that the dominant steady-state cost of an idle worker.
    hardware_refresh_interval_seconds: float = 300.0
    log_level: str = "info"
    # Phase 3 (schema 3): GPU utilization/memory counters are sampled
    # at most this often (in-process PDH; cheap, but there's no reason
    # to sample faster than anything consumes them). Heartbeats between
    # samples carry no GPU metrics, and the orchestrator keeps the
    # previous values.
    gpu_metrics_interval_seconds: float = 15.0
    # Run the small verified matmul test on each available inference
    # backend when profiling (see worker/hardware/backend_probe.py).
    backend_compute_test: bool = True
    # Path to a SIMULATED hardware profile YAML. Empty (the default) =
    # profile the real machine. When set, this worker process reports
    # the profile's hypothetical hardware, stamped SIMULATED, and never
    # inspects the real machine. See docs/SIMULATED_HARDWARE.md.
    simulated_profile: str = ""
    # Host/IP the orchestrator should dial back to reach this worker.
    # Empty = listen_host, or this machine's hostname when listening on
    # 0.0.0.0. On a Windows LAN where hostnames don't resolve between
    # machines, set this to the worker's LAN IP (see docs/NETWORKING.md).
    advertise_host: str = ""
    # Phase 6: the StageExecution service (worker/execution/). Off by
    # default; when on, it runs on its OWN gRPC server, separate from
    # the Health server above, bound to execution_listen_host -- loopback
    # unless configured. It is unauthenticated: bind it to a private-LAN
    # address only (docs/REMOTE_EXECUTION.md).
    execution_enabled: bool = False
    execution_listen_host: str = "127.0.0.1"
    execution_listen_port: int = 50200
    # 0.0.0.0 / :: are refused unless this is set (they include any
    # public interface the machine may have).
    execution_allow_all_interfaces: bool = False
    execution_max_tensor_mib: float = 16.0
    execution_max_message_mib: float = 32.0
    execution_max_concurrent: int = 2
    execution_max_batch: int = 8
    execution_max_sequence: int = 0  # 0 = the model's maximum positions
    execution_min_deadline_seconds: float = 0.05

    @property
    def listen_address(self) -> str:
        return f"{self.listen_host}:{self.listen_port}"


def default_worker_id() -> str:
    """A reasonable, locally-unique default -- callers (or worker.yaml,
    or WORKER_ID) should set a stable explicit ID for a real deployment
    so the same worker keeps its identity across restarts."""
    return f"{socket.gethostname()}-{os.getpid()}"


def load(config_path: Optional[str] = None, env_file: Optional[str] = ".env") -> WorkerConfig:
    """Load config starting from built-in defaults, then a YAML file (if
    it exists -- a missing file is not an error), then WORKER_* / env
    overrides, which always win.
    """
    load_dotenv(env_file)  # no-op if the file doesn't exist

    data: dict = {}
    if config_path:
        path = Path(config_path)
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

    cfg = WorkerConfig(
        worker_id=str(data.get("worker_id") or "") or default_worker_id(),
        orchestrator_address=str(data.get("orchestrator_address", "127.0.0.1:9090")),
        listen_host=str(data.get("listen_host", "127.0.0.1")),
        listen_port=int(data.get("listen_port", 50100)),
        heartbeat_interval_seconds=float(data.get("heartbeat_interval_seconds", 5.0)),
        hardware_refresh_interval_seconds=float(data.get("hardware_refresh_interval_seconds", 300.0)),
        log_level=str(data.get("log_level", "info")),
        gpu_metrics_interval_seconds=float(data.get("gpu_metrics_interval_seconds", 15.0)),
        backend_compute_test=_as_bool(data.get("backend_compute_test", True)),
        simulated_profile=str(data.get("simulated_profile") or ""),
        advertise_host=str(data.get("advertise_host") or ""),
        execution_enabled=_as_bool(data.get("execution_enabled", False)),
        execution_listen_host=str(data.get("execution_listen_host", "127.0.0.1")),
        execution_listen_port=int(data.get("execution_listen_port", 50200)),
        execution_allow_all_interfaces=_as_bool(data.get("execution_allow_all_interfaces", False)),
        execution_max_tensor_mib=float(data.get("execution_max_tensor_mib", 16.0)),
        execution_max_message_mib=float(data.get("execution_max_message_mib", 32.0)),
        execution_max_concurrent=int(data.get("execution_max_concurrent", 2)),
        execution_max_batch=int(data.get("execution_max_batch", 8)),
        execution_max_sequence=int(data.get("execution_max_sequence", 0)),
        execution_min_deadline_seconds=float(data.get("execution_min_deadline_seconds", 0.05)),
    )

    _apply_env_overrides(cfg)
    return cfg


def _apply_env_overrides(cfg: WorkerConfig) -> None:
    if value := os.environ.get("WORKER_ID"):
        cfg.worker_id = value
    if value := os.environ.get("WORKER_ORCHESTRATOR_ADDRESS"):
        cfg.orchestrator_address = value
    if value := os.environ.get("WORKER_LISTEN_HOST"):
        cfg.listen_host = value
    if value := os.environ.get("WORKER_LISTEN_PORT"):
        cfg.listen_port = int(value)
    if value := os.environ.get("WORKER_HEARTBEAT_INTERVAL_SECONDS"):
        cfg.heartbeat_interval_seconds = float(value)
    if value := os.environ.get("WORKER_HARDWARE_REFRESH_INTERVAL_SECONDS"):
        cfg.hardware_refresh_interval_seconds = float(value)
    if value := os.environ.get("WORKER_LOG_LEVEL"):
        cfg.log_level = value
    if value := os.environ.get("WORKER_GPU_METRICS_INTERVAL_SECONDS"):
        cfg.gpu_metrics_interval_seconds = float(value)
    if value := os.environ.get("WORKER_BACKEND_COMPUTE_TEST"):
        cfg.backend_compute_test = _as_bool(value)
    if value := os.environ.get("WORKER_SIMULATED_PROFILE"):
        cfg.simulated_profile = value
    if value := os.environ.get("WORKER_ADVERTISE_HOST"):
        cfg.advertise_host = value
    if value := os.environ.get("WORKER_EXECUTION_ENABLED"):
        cfg.execution_enabled = _as_bool(value)
    if value := os.environ.get("WORKER_EXECUTION_LISTEN_HOST"):
        cfg.execution_listen_host = value
    if value := os.environ.get("WORKER_EXECUTION_LISTEN_PORT"):
        cfg.execution_listen_port = int(value)
    if value := os.environ.get("WORKER_EXECUTION_ALLOW_ALL_INTERFACES"):
        cfg.execution_allow_all_interfaces = _as_bool(value)


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")
