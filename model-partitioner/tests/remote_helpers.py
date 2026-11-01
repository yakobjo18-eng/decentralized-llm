"""Shared helpers for the Phase 6 tests: two-stage plans for REAL-labelled
records of this host, in-process StageExecution servers, and separate
worker processes."""

from __future__ import annotations

import queue
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from conftest import REPO, real_local_worker, tiny_plan
from partitioner.execution import tiny_gpt2 as t
from partitioner.execution import wire
from partitioner.execution.interface import INPUT_STAGE, STAGE_INTERFACE_VERSION, Activation, ActivationMeta
from partitioner.memory import Workload

IDS = [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]]
WORKER_DIR = REPO / "worker"


def plan_for(worker_ids, **kw):
    kw.setdefault("workload", Workload(batch_size=2))
    kw.setdefault("allow_unused_workers", False)
    plan = tiny_plan(t.spec(), [real_local_worker(w) for w in worker_ids], **kw)
    assert len(plan["stages"]) == len(worker_ids), plan["stages"]
    return plan


def token_payload(ids=IDS):
    x = t.token_tensor(ids)
    return wire.encode(Activation(ActivationMeta(STAGE_INTERFACE_VERSION, "r", "token_ids", tuple(x.shape), "int64",
                                                 INPUT_STAGE), x))


def hidden_payload(batch=2, tokens=8, hidden=32, producer=0):
    import torch

    x = torch.zeros(batch, tokens, hidden)
    return wire.encode(Activation(ActivationMeta(STAGE_INTERFACE_VERSION, "r", "hidden_states", tuple(x.shape), "fp32",
                                                 producer), x))


class InProcessWorker:
    """The real servicer behind a real gRPC server on 127.0.0.1 -- in
    THIS process (unit-test speed). Not a separate worker process."""

    def __init__(self, worker_id, **cfg):
        from worker.execution.server import create_server
        from worker.execution.service import ServiceConfig, StageExecutionServicer

        hooks = {k: cfg.pop(k) for k in ("stage_factory", "before_compute") if k in cfg}
        self.servicer = StageExecutionServicer(ServiceConfig(worker_id=worker_id, **cfg), **hooks)
        self.server, port = create_server(self.servicer, "127.0.0.1", 0)
        self.server.start()
        self.endpoint = f"127.0.0.1:{port}"

    def stub(self):
        import grpc
        from worker.generated import stage_execution_pb2_grpc as pb_grpc

        self.channel = grpc.insecure_channel(self.endpoint)
        return pb_grpc.StageExecutionStub(self.channel)

    def stop(self):
        if getattr(self, "channel", None) is not None:
            self.channel.close()
        self.server.stop(grace=0).wait(2)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class WorkerProcess:
    """A separate OS process: `python -m worker.execution` (service only)
    or `python -m worker.main` (registration + heartbeats + service)."""

    def __init__(self, tmp: Path, worker_id: str, entry: str = "worker.execution", extra_yaml: str = "",
                 ready_marker: str = "STAGE_EXECUTION_LISTENING"):
        self.worker_id = worker_id
        config = tmp / f"{worker_id}.yaml"
        config.write_text(f'worker_id: "{worker_id}"\nexecution_listen_host: "127.0.0.1"\nlog_level: "info"\n' + extra_yaml)
        args = [sys.executable, "-m", entry, "--config", str(config)]
        if entry == "worker.execution":
            args += ["--port", "0"]
        self.proc = subprocess.Popen(args, cwd=WORKER_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                     creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        self.lines: "queue.Queue[str]" = queue.Queue()
        self.log = []
        threading.Thread(target=self._pump, daemon=True).start()
        self.ready_marker = ready_marker
        self.endpoint = None

    def _pump(self):
        for line in self.proc.stdout:
            self.log.append(line)
            self.lines.put(line)

    def wait_ready(self, timeout: float = 120.0) -> str:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                line = self.lines.get(timeout=0.5)
            except queue.Empty:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"{self.worker_id} exited early:\n{''.join(self.log)}")
                continue
            if self.ready_marker in line:
                if line.startswith("STAGE_EXECUTION_LISTENING"):
                    self.endpoint = line.split()[1]
                else:  # worker.main log line: "... listening on 127.0.0.1:PORT for worker ..."
                    self.endpoint = line.split(" listening on ")[1].split()[0]
                return self.endpoint
        raise TimeoutError(f"{self.worker_id} not ready in {timeout}s:\n{''.join(self.log)}")

    def kill(self):
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait(timeout=10)
