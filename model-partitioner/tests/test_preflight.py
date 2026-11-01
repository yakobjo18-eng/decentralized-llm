"""Phase 7 preflight diagnostics: runtime, worker config, TCP failure
classification, StageExecution endpoint checks (GetInfo, the Phase 6
fallback, wrong services), plan/assignment consistency, limits,
orchestrator checks, and the CLI."""

from __future__ import annotations

import http.server
import json
import socket
import threading
import time

import grpc
import pytest
from concurrent import futures

from conftest import REPO, real_local_worker, tiny_plan
from partitioner import preflight as pf
from partitioner.cli import main
from partitioner.execution import tiny_gpt2 as t
from remote_helpers import InProcessWorker, WorkerProcess, free_port, plan_for


def statuses(report, section_prefix="", name=None):
    return [c.status for c in report.checks if c.section.startswith(section_prefix) and (name is None or c.name == name)]


def one(report, section_prefix, name):
    found = [c for c in report.checks if c.section.startswith(section_prefix) and c.name == name]
    assert len(found) == 1, (section_prefix, name, [(c.section, c.name) for c in report.checks])
    return found[0]


def write(tmp_path, text, name="w.yaml"):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


GOOD = ('worker_id: "pf-a"\nlisten_host: "127.0.0.1"\nlisten_port: {hp}\nexecution_enabled: true\n'
        'execution_listen_host: "127.0.0.1"\nexecution_listen_port: {ep}\n')


@pytest.fixture
def good_config(tmp_path):
    return write(tmp_path, GOOD.format(hp=free_port(), ep=free_port()))


# ---- runtime and this machine ---------------------------------------------------


def test_runtime_checks_pass_here():
    r = pf.Report()
    pf.check_runtime(r)
    assert not r.failed, [c for c in r.checks if c.status == pf.FAIL]
    assert one(r, "runtime", "protocol stubs").status == pf.PASS
    assert {c.name for c in r.checks} >= {"python", "grpcio", "protobuf", "torch", "transformers", "numpy"}


def test_local_network_lists_real_interfaces():
    r = pf.Report()
    pf.check_local_network(r)
    assert one(r, "this machine", "hostname").detail == socket.gethostname()
    assert any(c.name == "IPv4 127.0.0.1" and c.detail.startswith("loopback") for c in r.checks)


# ---- worker config -----------------------------------------------------------------


def test_good_loopback_config(good_config):
    r = pf.Report()
    pf.check_worker_config(r, good_config)
    assert not r.failed and pf.WARN not in statuses(r)
    assert one(r, "worker config", "execution_listen_host").status == pf.INFO  # loopback: this machine only


@pytest.mark.parametrize("text, name, status, fragment", [
    ('worker_id: "a"\nlisten_host: "8.8.8.8"\n', "listen_host", pf.FAIL, "public address"),
    ('worker_id: "a"\nlisten_host: "my-pc"\n', "listen_host", pf.FAIL, "must be an IP address"),
    ('worker_id: "a"\nlisten_host: "0.0.0.0"\n', "listen_host", pf.WARN, "every interface"),
    ('worker_id: "a"\nexecution_enabled: true\nexecution_listen_host: "0.0.0.0"\n', "execution_listen_host", pf.FAIL,
     "EVERY interface"),
    ('worker_id: "a"\nexecution_enabled: true\nexecution_listen_host: "10.255.255.254"\n',
     "execution_listen_host ownership", pf.FAIL, "no interface with address 10.255.255.254"),
    ('worker_id: "a"\nexecution_enabled: true\nlisten_port: 50555\nexecution_listen_port: 50555\n', "ports", pf.FAIL,
     "both 50555"),
    ('worker_id: "a"\n', "execution_enabled", pf.WARN, "cannot run pipeline stages"),
    ('worker_id: "a"\nexecution_enabled: true\nsimulated_profile: "x.yaml"\n', "execution_enabled", pf.FAIL, "SIMULATED"),
    ('worker_id: "a"\norchestrator_address: "nonsense"\n', "orchestrator_address", pf.FAIL, "not HOST:PORT"),
    ('worker_id: "a"\nlisten_host: "192.168.77.77"\n', "listen_host ownership", pf.FAIL, "no interface"),
])
def test_bad_worker_configs(tmp_path, monkeypatch, text, name, status, fragment):
    for k in ("WORKER_LISTEN_HOST", "WORKER_EXECUTION_LISTEN_HOST", "WORKER_EXECUTION_ENABLED", "WORKER_ID"):
        monkeypatch.delenv(k, raising=False)
    r = pf.Report()
    pf.check_worker_config(r, write(tmp_path, text))
    c = one(r, "worker config", name)
    assert c.status == status and fragment in c.detail, (c.status, c.detail)
    if status == pf.FAIL:
        assert c.hint


def test_auto_generated_worker_id_and_missing_file(tmp_path, monkeypatch):
    monkeypatch.delenv("WORKER_ID", raising=False)
    r = pf.Report()
    pf.check_worker_config(r, write(tmp_path, "listen_port: 50101\n"))
    assert one(r, "worker config", "worker_id").status == pf.FAIL
    r = pf.Report()
    pf.check_worker_config(r, str(tmp_path / "nope.yaml"))
    assert one(r, "worker config", "file").status == pf.FAIL


def test_port_in_use_is_reported(tmp_path):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        port = s.getsockname()[1]
        r = pf.Report()
        pf.check_worker_config(r, write(tmp_path, GOOD.format(hp=free_port(), ep=port)))
    c = one(r, "worker config", f"execution_listen_host port {port}")
    assert c.status == pf.WARN and "in use" in c.detail


# ---- TCP failure classification ------------------------------------------------


def test_closed_port_is_refused_not_a_firewall_timeout():
    p = pf.tcp_probe("127.0.0.1", free_port(), timeout=3)
    assert (p.ok, p.kind) == (False, "refused")


def test_unresolvable_name_is_dns():
    p = pf.tcp_probe("no-such-host.invalid", 50200, timeout=2)
    assert (p.ok, p.kind) == (False, "dns") and "cannot resolve" in p.detail


def test_unroutable_address_is_timeout_or_unreachable():
    p = pf.tcp_probe("10.255.255.1", 50200, timeout=0.5)
    assert not p.ok and p.kind in ("timeout", "unreachable")
    assert "Firewall" in pf.hint_for(p) or "route" in pf.hint_for(p)


def test_loopback_timeout_hint_does_not_blame_the_firewall():
    p = pf.Probe(False, "timeout", "x", ip="127.0.0.1")
    assert "nothing is listening" in pf.hint_for(p)


# ---- endpoints -------------------------------------------------------------------


@pytest.fixture
def workers():
    started = []

    def start(wid, **kw):
        w = InProcessWorker(wid, **kw)
        started.append(w)
        return w

    yield start
    for w in started:
        w.stop()


def test_endpoint_with_getinfo(workers):
    w = workers("w-a")
    r = pf.Report()
    info = pf.check_endpoint(r, "w-a", w.endpoint, timeout=3)
    assert [c.status for c in r.checks if c.name in ("TCP reachable", "protocol", "identity")] == [pf.PASS] * 3
    assert info["protocol_version"] == 1 and info["authenticated"] is False
    assert info["executable_model_ids"] == ["builtin:tiny-gpt2-v1"] and info["worker_hostname"] == socket.gethostname()


def test_endpoint_identity_mismatch(workers):
    w = workers("w-b")
    r = pf.Report()
    pf.check_endpoint(r, "w-a", w.endpoint, timeout=3)
    c = one(r, "endpoint", "identity")
    assert c.status == pf.FAIL and "answers as worker 'w-b'" in c.detail


def test_incompatible_protocol_version():
    """A (hypothetical) future worker that only speaks protocol 2."""
    from worker.execution.server import create_server
    from worker.execution.service import ServiceConfig, StageExecutionServicer

    class FutureServicer(StageExecutionServicer):
        def GetInfo(self, request, context):  # noqa: N802
            resp = super().GetInfo(request, context)
            resp.protocol_version = 2
            del resp.supported_versions[:]
            resp.supported_versions.append(2)
            return resp

    server, port = create_server(FutureServicer(ServiceConfig(worker_id="w-a")), "127.0.0.1", 0)
    server.start()
    try:
        r = pf.Report()
        pf.check_endpoint(r, "w-a", f"127.0.0.1:{port}", timeout=3)
        c = one(r, "endpoint", "protocol")
        assert c.status == pf.FAIL and "worker speaks [2], coordinator speaks 1" in c.detail and c.hint
    finally:
        server.stop(0)


def test_phase6_worker_without_getinfo_is_probed_safely():
    """A worker from before GetInfo: answered via a version-0 ExecuteStage
    that is rejected before any execution."""
    from worker.execution.service import ServiceConfig, StageExecutionServicer
    from worker.generated import stage_execution_pb2_grpc as pb_grpc

    class Phase6Servicer(StageExecutionServicer):
        GetInfo = pb_grpc.StageExecutionServicer.GetInfo  # the generated UNIMPLEMENTED stub

    calls = []
    servicer = Phase6Servicer(ServiceConfig(worker_id="old"), stage_factory=lambda *a: calls.append(a))
    server = grpc.server(futures.ThreadPoolExecutor(2))
    pb_grpc.add_StageExecutionServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        r = pf.Report()
        info = pf.check_endpoint(r, "old", f"127.0.0.1:{port}", timeout=3)
        c = one(r, "endpoint", "protocol")
        assert c.status == pf.WARN and "predates GetInfo" in c.detail
        assert info["probe_error"] == "PROTOCOL_VERSION" and calls == []  # nothing was built or executed
        assert one(r, "endpoint", "identity").status == pf.PASS
    finally:
        server.stop(0)


def test_port_that_is_not_stage_execution():
    server = grpc.server(futures.ThreadPoolExecutor(1))  # a gRPC server without StageExecution (like Health)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        r = pf.Report()
        assert pf.check_endpoint(r, "x", f"127.0.0.1:{port}", timeout=3) is None
        c = one(r, "endpoint", "protocol")
        assert c.status == pf.FAIL and "UNIMPLEMENTED" in c.detail and "Health port" in c.hint
    finally:
        server.stop(0)


def test_plain_tcp_listener_is_not_grpc():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        port = s.getsockname()[1]
        r = pf.Report()
        pf.check_endpoint(r, "x", f"127.0.0.1:{port}", timeout=1.5)
    assert one(r, "endpoint", "TCP reachable").status == pf.PASS
    assert one(r, "endpoint", "protocol").status == pf.FAIL


def test_unreachable_endpoint_is_not_called_reachable():
    r = pf.Report()
    assert pf.check_endpoint(r, "w", f"127.0.0.1:{free_port()}", timeout=3) is None
    assert one(r, "endpoint", "TCP reachable").status == pf.FAIL
    assert one(r, "endpoint", "protocol").status == pf.SKIP


# ---- plan --------------------------------------------------------------------------


def run_plan(tmp_path, plan, endpoints, **kw):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    return pf.run(endpoints=endpoints, plan=str(path), skip_runtime=True, timeout=3, **kw)


def test_consistent_plan_passes(tmp_path, workers):
    a, b = workers("w-a"), workers("w-b")
    r = run_plan(tmp_path, plan_for(["w-a", "w-b"]), {"w-a": a.endpoint, "w-b": b.endpoint})
    assert not r.failed, [(c.name, c.detail) for c in r.checks if c.status == pf.FAIL]
    assert statuses(r, "plan", "stage 0 -> w-a") == [pf.PASS] and statuses(r, "plan", "limits stage 1") == [pf.PASS]
    assert one(r, "plan", "hosts spanned").detail.startswith("1 host")


def test_swapped_and_missing_endpoints_fail(tmp_path, workers):
    a, b = workers("w-a"), workers("w-b")
    r = run_plan(tmp_path, plan_for(["w-a", "w-b"]), {"w-a": b.endpoint, "w-b": a.endpoint})
    assert statuses(r, "plan", "stage 0 -> w-a") == [pf.FAIL]
    r = run_plan(tmp_path, plan_for(["w-a", "w-b"]), {"w-a": a.endpoint})
    c = one(r, "plan", "stage 1 -> w-b")
    assert c.status == pf.FAIL and "no --endpoint" in c.detail


def test_multi_host_requirement(tmp_path, workers):
    a, b = workers("w-a"), workers("w-b")
    r = run_plan(tmp_path, plan_for(["w-a", "w-b"]), {"w-a": a.endpoint, "w-b": b.endpoint}, require_multi_host=True)
    assert one(r, "plan", "hosts spanned").status == pf.FAIL


def test_loopback_endpoint_for_a_remote_host_fails(tmp_path, workers):
    a = workers("w-a")
    plan = tiny_plan(t.spec(), [real_local_worker("w-a"), real_local_worker("c2", hostname="computer-2")],
                     allow_unused_workers=False)
    r = run_plan(tmp_path, plan, {"w-a": a.endpoint, "c2": f"127.0.0.1:{free_port()}"})
    c = one(r, "plan", [c.name for c in r.checks if c.name.endswith("-> c2")][0])
    assert c.status == pf.FAIL and "loopback but the stage is planned for host 'computer-2'" in c.detail


def test_worker_limits_too_small_fail(tmp_path, workers):
    a, b = workers("w-a"), workers("w-b", max_tensor_bytes=4096, max_batch=1)
    r = run_plan(tmp_path, plan_for(["w-a", "w-b"]), {"w-a": a.endpoint, "w-b": b.endpoint})
    c = one(r, "plan", "limits stage 1")
    assert c.status == pf.FAIL and "max_tensor_bytes 4096" in c.detail and "max_batch 1" in c.detail


def test_invalid_plan_fails(tmp_path):
    plan = plan_for(["w-a", "w-b"])
    plan["workload"]["prompt_length"] = 3  # edited after planning
    r = run_plan(tmp_path, plan, {})
    assert any(c.status == pf.FAIL and "modified after planning" in c.detail for c in r.checks)


# ---- orchestrator (stand-in HTTP server: SIMULATED) ----------------------------------


def test_orchestrator_checks_against_a_stand_in_server():
    workers_json = [
        {"id": "ok", "online": True, "profile_source": "REAL", "hostname": "h1", "address": "127.0.0.1:1",
         "last_heartbeat_age_seconds": 0.5},
        {"id": "off", "online": False, "profile_source": "REAL", "hostname": "h2", "address": "127.0.0.1:2",
         "last_heartbeat_age_seconds": 40.0},
        {"id": "sim", "online": True, "profile_source": "SIMULATED", "hostname": "h3", "address": "127.0.0.1:3",
         "last_heartbeat_age_seconds": 0.2},
    ]

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            body = {"/health": {"status": "ok"}, "/api/workers": workers_json}.get(self.path)
            self.send_response(200 if body is not None else 404)
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def log_message(self, *a):
            pass

    httpd = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        r = pf.Report()
        pf.check_orchestrator(r, f"http://127.0.0.1:{httpd.server_port}", ["ok", "off", "sim", "missing"], timeout=3)
        assert one(r, "orchestrator", "health").status == pf.PASS
        assert one(r, "orchestrator", "worker ok").status == pf.PASS
        assert "offline" in one(r, "orchestrator", "worker off").detail
        assert "SIMULATED, not REAL" in one(r, "orchestrator", "worker sim").detail
        assert one(r, "orchestrator", "worker missing").status == pf.FAIL
    finally:
        httpd.shutdown()


# ---- CLI -----------------------------------------------------------------------


def test_cli_exit_codes(capsys, good_config):
    assert main(["preflight", "--skip-runtime", "--worker-config", good_config]) == 0
    assert "no failures" in capsys.readouterr().out
    assert main(["preflight", "--skip-runtime", "--endpoint", f"w=127.0.0.1:{free_port()}", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is False and report["counts"]["FAIL"] >= 1
    assert main(["preflight", "--endpoint", "garbage"]) == 2


# ---- live: real orchestrator + worker processes (REAL_MULTI_PROCESS) ----------------


ORCH_EXE = REPO / "orchestrator" / "orchestrator.exe"


@pytest.mark.skipif(not ORCH_EXE.exists(), reason="orchestrator.exe not built")
def test_preflight_against_a_live_orchestrator_and_worker_processes(tmp_path):
    import subprocess
    import urllib.request

    rest, grpc_port = free_port(), free_port()
    (tmp_path / "o.yaml").write_text(f'rest_address: "127.0.0.1:{rest}"\ngrpc_address: "127.0.0.1:{grpc_port}"\n')
    orch = subprocess.Popen([str(ORCH_EXE), "--config", str(tmp_path / "o.yaml")],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    procs = {}
    try:
        for wid in ("lp-a", "lp-b"):
            extra = (f'orchestrator_address: "127.0.0.1:{grpc_port}"\nlisten_port: {free_port()}\n'
                     'heartbeat_interval_seconds: 0.5\nbackend_compute_test: false\nexecution_enabled: true\n'
                     'execution_listen_port: 0\n')
            procs[wid] = WorkerProcess(tmp_path, wid, entry="worker.main", extra_yaml=extra,
                                       ready_marker="StageExecution (protocol v1, UNAUTHENTICATED) listening on")
        eps = {wid: p.wait_ready() for wid, p in procs.items()}
        url = f"http://127.0.0.1:{rest}"
        # Online AND backend probe finished: until then preflight correctly WARNs that planning can't use the worker.
        deadline = time.time() + 120
        while time.time() < deadline:
            try:
                ws = json.loads(urllib.request.urlopen(f"{url}/api/workers", timeout=2).read())
                if {w["id"] for w in ws if w["online"] and not w.get("backend_probe_pending")} >= set(procs):
                    break
            except OSError:
                pass
            time.sleep(0.3)
        else:
            pytest.fail("workers did not register and finish their backend probes in time")
        path = tmp_path / "plan.json"
        path.write_text(json.dumps(plan_for(["lp-a", "lp-b"])))
        r = pf.run(orchestrator=url, endpoints=eps, plan=str(path), timeout=3, skip_runtime=True)
        failures = [(c.section, c.name, c.detail) for c in r.checks if c.status == pf.FAIL]
        assert not failures, failures
        assert one(r, "orchestrator", "worker lp-a").status == pf.PASS
        assert one(r, "plan", "stage 1 -> lp-b").status == pf.PASS
        assert "vs. orchestrator" in one(r, "plan", "stage 0 -> lp-a").tested
    finally:
        for p in procs.values():
            p.kill()
        orch.kill()
        orch.wait(timeout=10)
