"""Integration test: starts a minimal fake WorkerRegistry gRPC server
in-process and drives RegistryClient against it. This proves the
generated Python stubs and the client's message construction are wire-
compatible with shared/proto/registry.proto -- the same contract the Go
orchestrator implements for real in
orchestrator/internal/grpcserver/registry_server.go.
"""

from __future__ import annotations

from concurrent import futures

import grpc
import pytest

from worker.communication.registry_client import RegistryClient
from worker.generated import registry_pb2, registry_pb2_grpc
from worker.hardware.profiler import HardwareSnapshot


class _FakeRegistryServicer(registry_pb2_grpc.WorkerRegistryServicer):
    def __init__(self):
        self.registered: dict[str, registry_pb2.RegisterRequest] = {}

    def Register(self, request, context):  # noqa: N802
        self.registered[request.worker_id] = request
        return registry_pb2.RegisterResponse(accepted=True, message="registered")

    def Heartbeat(self, request, context):  # noqa: N802
        ok = request.worker_id in self.registered
        return registry_pb2.HeartbeatResponse(acknowledged=ok)

    def Deregister(self, request, context):  # noqa: N802
        existed = self.registered.pop(request.worker_id, None) is not None
        return registry_pb2.DeregisterResponse(accepted=existed)


@pytest.fixture()
def fake_server():
    servicer = _FakeRegistryServicer()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    registry_pb2_grpc.add_WorkerRegistryServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    yield servicer, f"127.0.0.1:{port}"
    server.stop(grace=1)


def _hardware(**overrides) -> HardwareSnapshot:
    base = dict(
        hostname="test-host",
        cpu_logical_cores=4,
        total_ram_bytes=1 << 30,
        has_cuda_gpu=False,
        gpu_name="",
        gpu_vram_bytes=0,
        os="Windows",
        python_version="3.13.3",
    )
    base.update(overrides)
    return HardwareSnapshot(**base)


def test_register_round_trip(fake_server):
    servicer, address = fake_server
    client = RegistryClient(address)

    response = client.register("worker-x", "127.0.0.1:50100", _hardware(cpu_logical_cores=4))

    assert response.accepted
    assert "worker-x" in servicer.registered
    assert servicer.registered["worker-x"].hardware.cpu_logical_cores == 4
    client.close()


def test_heartbeat_and_deregister_round_trip(fake_server):
    servicer, address = fake_server
    client = RegistryClient(address)
    client.register("worker-y", "127.0.0.1:50101", _hardware())

    heartbeat = client.heartbeat("worker-y", status="busy")
    assert heartbeat.acknowledged

    deregister = client.deregister("worker-y")
    assert deregister.accepted
    assert "worker-y" not in servicer.registered
    client.close()


def test_heartbeat_for_unknown_worker_not_acknowledged(fake_server):
    _servicer, address = fake_server
    client = RegistryClient(address)

    heartbeat = client.heartbeat("ghost")

    assert not heartbeat.acknowledged
    client.close()
