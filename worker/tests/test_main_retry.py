"""Unit tests for worker.main._register_with_retry: the Phase 2
startup retry-with-backoff logic, isolated from any real network call
via a fake registry client."""

from __future__ import annotations

import threading
import time

import grpc

from worker import main as worker_main
from worker.config.loader import WorkerConfig


class _FlakyThenOK:
    """Fails with a transport-level error `fail_times` times, then
    succeeds."""

    def __init__(self, fail_times: int):
        self.calls = 0
        self.fail_times = fail_times

    def register(self, worker_id, own_address, hardware):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise grpc.RpcError("simulated: orchestrator unreachable")
        return "ok-response"


class _AlwaysFlaky:
    def __init__(self):
        self.calls = 0

    def register(self, worker_id, own_address, hardware):
        self.calls += 1
        raise grpc.RpcError("simulated: orchestrator unreachable")


def _cfg() -> WorkerConfig:
    return WorkerConfig(worker_id="w1", orchestrator_address="127.0.0.1:9090")


def test_register_with_retry_succeeds_after_transient_failures():
    client = _FlakyThenOK(fail_times=2)
    stop_event = threading.Event()

    result = worker_main._register_with_retry(
        client, _cfg(), hardware=None, own_address="x",
        stop_event=stop_event, initial_delay=0.01, max_delay=0.01,
    )

    assert result == "ok-response"
    assert client.calls == 3


def test_register_with_retry_stops_promptly_on_stop_event():
    client = _AlwaysFlaky()
    stop_event = threading.Event()

    def stopper():
        time.sleep(0.05)
        stop_event.set()

    t = threading.Thread(target=stopper)
    t.start()

    start = time.monotonic()
    result = worker_main._register_with_retry(
        client, _cfg(), hardware=None, own_address="x",
        stop_event=stop_event, initial_delay=0.02, max_delay=0.02,
    )
    elapsed = time.monotonic() - start
    t.join()

    assert result is None
    assert client.calls >= 1
    # Should stop close to when stop_event was set, not hang.
    assert elapsed < 1.0


def test_register_with_retry_does_not_retry_an_application_level_rejection():
    class _RejectsImmediately:
        def __init__(self):
            self.calls = 0

        def register(self, worker_id, own_address, hardware):
            self.calls += 1
            return "rejected-response"  # no exception -- a real (if unaccepted) reply

    client = _RejectsImmediately()
    stop_event = threading.Event()

    result = worker_main._register_with_retry(
        client, _cfg(), hardware=None, own_address="x",
        stop_event=stop_event, initial_delay=5.0, max_delay=5.0,
    )

    assert result == "rejected-response"
    assert client.calls == 1  # returned immediately, no retry loop entered
