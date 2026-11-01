"""gRPC Health service, implemented by the worker (not the
orchestrator) so the orchestrator -- or an operator, or a test -- can
directly query this specific worker's live status, as distinct from
the orchestrator-side registry's last-known-heartbeat view.
"""

from __future__ import annotations

import time
from concurrent import futures

import grpc

from worker.generated import health_pb2, health_pb2_grpc


class HealthServicer(health_pb2_grpc.HealthServicer):
    def __init__(self, worker_id: str):
        self._worker_id = worker_id

    def Check(self, request, context):  # noqa: N802 -- gRPC-generated method name
        return health_pb2.HealthCheckResponse(
            status="SERVING",
            worker_id=self._worker_id,
            server_time_unix_ms=int(time.time() * 1000),
        )


def create_server(worker_id: str, listen_address: str, max_workers: int = 4) -> grpc.Server:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    health_pb2_grpc.add_HealthServicer_to_server(HealthServicer(worker_id), server)
    server.add_insecure_port(listen_address)
    return server
