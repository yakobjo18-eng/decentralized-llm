"""Worker node entrypoint.

Usage:
    python -m worker.main --config worker.yaml

Phase 1 scope (real, working, end to end):
  - Load configuration (worker.yaml + WORKER_* env overrides).
  - Collect a minimal, real hardware snapshot (worker/hardware/profiler.py).
  - Start this worker's own gRPC Health service.
  - Register with the orchestrator over gRPC (WorkerRegistry.Register).
  - Send periodic heartbeats (WorkerRegistry.Heartbeat).
  - Deregister cleanly on shutdown (Ctrl+C / SIGINT).

Phase 2 adds:
  - Retry-with-backoff on the INITIAL connection to the orchestrator
    (previously a single failed Register() call crashed the process
    immediately) -- see _register_with_retry. The heartbeat loop
    already retried on failure in Phase 1; this closes the same gap at
    startup, satisfying "handle temporary network disconnections" for
    the case where the worker starts before the orchestrator is up.
  - SIGBREAK handling (Windows-only; a no-op import on other
    platforms), so an operator or test harness can request a graceful
    shutdown of one specific background worker process via
    os.kill(pid, signal.CTRL_BREAK_EVENT) without force-killing it --
    see docs/RUNNING_PHASE2.md and scripts/multi_worker_test.py.
  - Clearer startup/connection/rejection logging (objective 2).

Phase 3 adds:
  - Every heartbeat now carries cheap live CPU/RAM metrics
    (collect_live_metrics(), psutil only -- no GPU re-detection).
  - Every hardware_refresh_interval_seconds (default 60s at the time;
    300s since schema 3, much less
    frequent than heartbeats), the worker re-collects its FULL
    hardware snapshot (collect(), including GPU enumeration) and
    re-registers -- this is how "allow workers to refresh their
    hardware snapshots" (objective 7) is implemented: reusing
    Register(), which Phase 2 already made safe to call again from the
    same address, rather than adding a fourth RPC. See
    docs/HARDWARE_PROFILING.md.

Phase 3 (schema 3) adds:
  - A HardwareProfiler that caches the static profile and samples GPU
    counters on their own interval (gpu_metrics_interval_seconds).
  - Reconnection handling: if the orchestrator answers a heartbeat
    with reregister_required (it restarted and lost its in-memory
    registry), or heartbeats start succeeding again after failing
    (a network outage ended), the worker re-collects its full hardware
    profile and re-registers. Before this, a restarted orchestrator
    never saw the worker again until the worker itself was restarted.
  - simulated_profile: run this worker process with a SIMULATED
    hardware profile instead of the real machine's (see
    docs/SIMULATED_HARDWARE.md).

Phase 6 adds:
  - When execution_enabled is true, the StageExecution service
    (worker/execution/) starts on its OWN gRPC server and port
    (execution_listen_host:execution_listen_port, loopback by default),
    separate from the Health server. Registration and heartbeats are
    unchanged: the orchestrator is not told about the service. See
    docs/REMOTE_EXECUTION.md.
"""

from __future__ import annotations

import argparse
import logging
import signal
import socket
import threading
import time

import grpc

from worker.communication.health_server import create_server
from worker.communication.registry_client import RegistryClient
from worker.config.loader import WorkerConfig, load as load_config
from worker.hardware.profiler import HardwareProfiler
from worker.hardware.simulated import SIMULATED_NOTICE, load_profile

logger = logging.getLogger("worker")


def _build_profiler(cfg: WorkerConfig) -> HardwareProfiler:
    if cfg.simulated_profile:
        profile = load_profile(cfg.simulated_profile)
        logger.warning(
            "%s: this worker will report the simulated profile %r from %s, "
            "NOT the hardware of the machine it is running on",
            SIMULATED_NOTICE,
            profile.name,
            cfg.simulated_profile,
        )
        return HardwareProfiler(simulated_snapshot=profile.snapshot, simulated_live=profile.live)
    return HardwareProfiler(
        gpu_metrics_interval_seconds=cfg.gpu_metrics_interval_seconds,
        run_compute_test=cfg.backend_compute_test,
        isolated_backend_probe=True,
    )


def _reregister(registry_client, cfg: WorkerConfig, profiler: HardwareProfiler, own_address: str, reason: str) -> bool:
    """Re-collects the static profile (CPU, RAM, GPU inventory) plus
    fresh live metrics and registers again. Returns True if the
    orchestrator accepted it. Never raises.

    The backend probe is never re-run here: what a framework can
    compute on doesn't change while the process runs, and re-probing
    (8-14 s) would stall heartbeats. If the startup probe is still
    running, the profile goes out marked backend_probe_pending."""
    try:
        profiler.refresh_static(reprobe_backends=False, wait_for_backends=False)
        response = registry_client.register(cfg.worker_id, own_address, profiler.full_snapshot())
    except Exception:
        logger.warning("re-registration (%s) failed", reason, exc_info=True)
        return False
    if response.accepted:
        logger.info("re-registered with fresh hardware profile (%s): %s", reason, response.message)
        return True
    # Should not normally happen -- same worker_id, same address as the
    # original registration, so Phase 2's conflict check treats this as
    # a refresh, not a conflict. Logged in case it ever does (e.g. the
    # orchestrator restarted and lost state, then a different worker
    # grabbed this ID in the meantime).
    logger.warning("re-registration (%s) rejected: %s", reason, response.message)
    return False


def _own_advertised_address(cfg: WorkerConfig) -> str:
    """Advertise this machine's real hostname instead of 0.0.0.0 so an
    orchestrator on a different machine can actually dial back to this
    worker's gRPC server. advertise_host (Phase 3), if set, wins -- for
    LANs where the hostname doesn't resolve from other machines."""
    if cfg.advertise_host:
        return f"{cfg.advertise_host}:{cfg.listen_port}"
    host = cfg.listen_host
    if host in ("0.0.0.0", "::"):
        host = socket.gethostname()
    return f"{host}:{cfg.listen_port}"


def _register_with_retry(
    registry_client,
    cfg: WorkerConfig,
    hardware,
    own_address: str,
    stop_event: threading.Event,
    initial_delay: float = 2.0,
    max_delay: float = 10.0,
):
    """Attempt registration, retrying with backoff while the
    orchestrator is unreachable, until it succeeds or stop_event is set
    (e.g. by Ctrl+C while still waiting to connect). Returns the
    RegisterResponse, or None if shutdown was requested before a
    connection succeeded.

    This only covers transport-level failures (orchestrator not up
    yet, network unreachable). A response that comes back but is
    rejected (response.accepted is False, e.g. a duplicate worker ID
    conflict) is returned immediately, not retried -- retrying an
    application-level rejection would just get the same rejection
    again.
    """
    attempt = 0
    delay = initial_delay
    while not stop_event.is_set():
        attempt += 1
        try:
            logger.info(
                "connecting to orchestrator at %s (attempt %d)...",
                cfg.orchestrator_address,
                attempt,
            )
            return registry_client.register(cfg.worker_id, own_address, hardware)
        except grpc.RpcError as exc:
            logger.warning(
                "orchestrator unreachable at %s (%s); retrying in %.1fs",
                cfg.orchestrator_address,
                exc,
                delay,
            )
            stop_event.wait(delay)
            delay = min(delay * 1.5, max_delay)
    return None


def run(config_path: str) -> None:
    cfg = load_config(config_path)
    logging.basicConfig(
        level=cfg.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info(
        "starting worker %s (orchestrator=%s, listen=%s, heartbeat_interval=%.1fs)",
        cfg.worker_id,
        cfg.orchestrator_address,
        cfg.listen_address,
        cfg.heartbeat_interval_seconds,
    )

    # Signal handling is set up before attempting registration so that
    # Ctrl+C (or a graceful CTRL_BREAK_EVENT on Windows, used by
    # scripts/multi_worker_test.py) works even while still retrying the
    # initial connection to the orchestrator, not just once the main
    # heartbeat loop has started.
    stop_event = threading.Event()

    def _handle_signal(signum, frame):  # noqa: ANN001 -- signal handler signature
        logger.info("signal %s received, shutting down", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    for signal_name in ("SIGTERM", "SIGBREAK"):
        sig = getattr(signal, signal_name, None)
        if sig is not None:
            try:
                signal.signal(sig, _handle_signal)
            except (AttributeError, ValueError, OSError):
                pass  # not meaningfully available in this context/platform

    health_server = create_server(cfg.worker_id, cfg.listen_address)
    health_server.start()
    logger.info("health gRPC service listening on %s", cfg.listen_address)
    execution_server = None
    if cfg.execution_enabled:
        from worker.execution.server import start_from_config

        try:
            execution_server, _ = start_from_config(cfg)
        except Exception:
            health_server.stop(grace=1)
            raise

    def _stop_servers(grace: float) -> None:
        if execution_server is not None:
            execution_server.stop(grace=grace).wait(grace + 1)
        health_server.stop(grace=grace).wait(grace + 1)

    registry_client = RegistryClient(cfg.orchestrator_address)
    own_address = _own_advertised_address(cfg)
    profiler = _build_profiler(cfg)
    # Register with the fast part of the profile first (~1.5 s); the
    # backend probe (`import torch` + compute test in a child process,
    # 8-14 s on Computer 1) runs in the background and triggers a
    # re-registration when it finishes. Waiting for it made three
    # workers starting together take 23-26 s to appear.
    profiler.refresh_static(wait_for_backends=False)
    hardware = profiler.full_snapshot()
    logger.info("hardware snapshot (%s): %s", hardware.profile_source, hardware)
    backends_ready = threading.Event()
    if not profiler.backends_ready:

        def _probe_backends() -> None:
            started = time.monotonic()
            profiler.probe_backends()
            logger.info("inference backend probe finished in %.1fs", time.monotonic() - started)
            backends_ready.set()

        threading.Thread(target=_probe_backends, name="backend-probe", daemon=True).start()

    response = _register_with_retry(registry_client, cfg, hardware, own_address, stop_event)
    if response is None:
        logger.info("shutdown requested before a connection to the orchestrator succeeded")
        registry_client.close()
        profiler.close()
        _stop_servers(1)
        return

    if not response.accepted:
        logger.error(
            "registration rejected by orchestrator: %s "
            "(this usually means worker_id=%s is already registered and active "
            "from a different address -- set a different worker_id, or wait for "
            "the previous registration to time out if it's actually gone)",
            response.message,
            cfg.worker_id,
        )
        registry_client.close()
        profiler.close()
        _stop_servers(1)
        return
    logger.info("registered with orchestrator: %s", response.message)

    last_hardware_refresh = time.monotonic()
    consecutive_failures = 0
    try:
        while not stop_event.is_set():
            stop_event.wait(cfg.heartbeat_interval_seconds)
            if stop_event.is_set():
                break
            try:
                live = profiler.live_metrics()
                hb = registry_client.heartbeat(cfg.worker_id, status="idle", live_metrics=live)
                logger.debug(
                    "heartbeat acknowledged=%s cpu=%s%% ram=%s%% gpu_samples=%d",
                    hb.acknowledged,
                    live.cpu_utilization_percent,
                    live.ram_utilization_percent,
                    len(live.gpu_metrics),
                )
                if consecutive_failures:
                    # Connectivity is back after an outage. The machine may
                    # have changed meanwhile (e.g. memory freed), so send a
                    # fresh full profile rather than trusting the old one.
                    logger.info("orchestrator reachable again after %d failed heartbeats", consecutive_failures)
                    consecutive_failures = 0
                    if _reregister(registry_client, cfg, profiler, own_address, "reconnected"):
                        last_hardware_refresh = time.monotonic()
                elif not hb.acknowledged:
                    if getattr(hb, "reregister_required", False):
                        logger.warning("orchestrator no longer knows this worker (it may have restarted)")
                        if _reregister(registry_client, cfg, profiler, own_address, "orchestrator lost registration"):
                            last_hardware_refresh = time.monotonic()
                    else:
                        logger.warning("heartbeat rejected: %s", getattr(hb, "message", ""))
            except Exception:
                # Keep trying -- a transient network blip should not crash
                # the worker. The orchestrator will mark this worker
                # offline after heartbeat_timeout_seconds if the outage
                # persists (Phase 2); once connectivity returns, the next
                # successful heartbeat here brings it back online there.
                consecutive_failures += 1
                logger.warning("heartbeat failed (%d in a row)", consecutive_failures, exc_info=True)

            if backends_ready.is_set() and consecutive_failures == 0:
                backends_ready.clear()
                if _reregister(registry_client, cfg, profiler, own_address, "backend probe complete"):
                    last_hardware_refresh = time.monotonic()

            now = time.monotonic()
            if consecutive_failures == 0 and now - last_hardware_refresh >= cfg.hardware_refresh_interval_seconds:
                last_hardware_refresh = now
                # Inventory only: the backend probe (torch + compute
                # test) is reused, see HardwareProfiler.refresh_static.
                _reregister(registry_client, cfg, profiler, own_address, "periodic refresh")
    finally:
        try:
            registry_client.deregister(cfg.worker_id)
        except Exception:
            logger.warning("deregister failed", exc_info=True)
        registry_client.close()
        profiler.close()
        _stop_servers(2)
        logger.info("worker shutdown complete")


def main() -> None:
    parser = argparse.ArgumentParser(description="Decentralized LLM inference worker node")
    parser.add_argument(
        "--config",
        default="worker.yaml",
        help="Path to worker config YAML (a missing file is fine; defaults + env vars still apply)",
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
