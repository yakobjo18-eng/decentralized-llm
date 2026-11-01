"""The StageExecution gRPC server: its own server and port, separate from
the worker's Health server, with explicit limits.

Bind-address policy (the protocol is unauthenticated): see
worker/config/bind_policy.py. 0.0.0.0 / :: additionally needs
execution_allow_all_interfaces: true.
"""

from __future__ import annotations

import logging
from concurrent import futures
from typing import Optional, Tuple

import grpc

from worker.config.bind_policy import BindAddressError, check_bind_host  # noqa: F401 -- re-exported
from worker.config.loader import WorkerConfig
from worker.execution.service import MIB, ServiceConfig, StageExecutionServicer, ensure_partitioner_importable
from worker.generated import stage_execution_pb2_grpc as pb_grpc

logger = logging.getLogger("worker.execution")


def service_config(cfg: WorkerConfig) -> ServiceConfig:
    return ServiceConfig(
        worker_id=cfg.worker_id,
        max_tensor_bytes=int(cfg.execution_max_tensor_mib * MIB),
        max_message_bytes=int(cfg.execution_max_message_mib * MIB),
        max_concurrent=cfg.execution_max_concurrent,
        max_batch=cfg.execution_max_batch,
        max_sequence=cfg.execution_max_sequence,
        min_deadline_seconds=cfg.execution_min_deadline_seconds,
    )


def create_server(servicer: StageExecutionServicer, host: str, port: int) -> Tuple[grpc.Server, int]:
    """Returns (server, bound port); port 0 picks a free port."""
    c = servicer.cfg
    options = [
        ("grpc.max_receive_message_length", c.max_message_bytes),
        ("grpc.max_send_message_length", c.max_message_bytes),
    ]
    # Threads beyond max_concurrent answer BUSY quickly; gRPC itself
    # refuses (RESOURCE_EXHAUSTED) anything past maximum_concurrent_rpcs.
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=c.max_concurrent + 2, thread_name_prefix="stage-exec"),
        options=options,
        maximum_concurrent_rpcs=c.max_concurrent + 2,
    )
    pb_grpc.add_StageExecutionServicer_to_server(servicer, server)
    bound = server.add_insecure_port(f"{host}:{port}")
    if not bound:
        raise OSError(f"could not bind the StageExecution server to {host}:{port}")
    return server, bound


def start_from_config(cfg: WorkerConfig, port: Optional[int] = None) -> Tuple[grpc.Server, str]:
    """Validates the configuration, pre-loads the executable model's
    spec (so the first request isn't slowed by importing transformers),
    and starts the server. Returns (server, "host:port")."""
    if cfg.simulated_profile:
        raise RuntimeError("refusing to start StageExecution: this worker runs a SIMULATED hardware profile, so it "
                           "is not the machine it claims to be")
    check_bind_host(cfg.execution_listen_host, cfg.execution_allow_all_interfaces)
    ensure_partitioner_importable()
    from partitioner.execution.models import EXECUTABLE_MODEL_IDS, get_executable_model

    for model_id in EXECUTABLE_MODEL_IDS:
        get_executable_model(model_id).spec()
    servicer = StageExecutionServicer(service_config(cfg))
    server, bound = create_server(servicer, cfg.execution_listen_host,
                                  cfg.execution_listen_port if port is None else port)
    server.start()
    address = f"{cfg.execution_listen_host}:{bound}"
    logger.info("StageExecution (protocol v1, UNAUTHENTICATED) listening on %s for worker %s; max tensor %d B, "
                "max message %d B, max concurrent %d", address, cfg.worker_id, servicer.cfg.max_tensor_bytes,
                servicer.cfg.max_message_bytes, servicer.cfg.max_concurrent)
    return server, address
