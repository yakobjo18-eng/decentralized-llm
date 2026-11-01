"""Phase 8: two-machine LAN workflow -- firewall check, orchestrator config
check, worker-side orchestrator probe, config generators, the
multi-machine verdict, Go/Python bind-policy parity, and full loopback
rehearsals of `scripts/phase8_lan.py run`. Nothing here involves a second
computer: the rehearsals are REAL_MULTI_PROCESS and must say so."""

from __future__ import annotations

import importlib.util
import json
import socket
import subprocess
import sys

import pytest

from conftest import REPO
from partitioner import preflight as pf
from remote_helpers import free_port

ORCH_EXE = REPO / "orchestrator" / "orchestrator.exe"
needs_orch = pytest.mark.skipif(not ORCH_EXE.exists(), reason="orchestrator.exe not built")


def load_phase8():
    spec = importlib.util.spec_from_file_location("phase8_lan", REPO / "scripts" / "phase8_lan.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P8 = load_phase8()


def one(report, prefix, name):
    found = [c for c in report.checks if c.section.startswith(prefix) and c.name == name]
    assert len(found) == 1, [(c.section, c.name) for c in report.checks]
    return found[0]


# ---- firewall evaluation on canned netsh output (SIMULATED firewall state) -------------

PROFILE_ON = "Private Profile Settings: \n----------------------------------------------------------------------\nState                                 ON\n"
PROFILE_OFF = PROFILE_ON.replace("ON\n", "OFF\n")


def rule(name, port="50201", action="Allow", remote="Any", program="Any", profiles="Private", enabled="Yes",
         protocol="TCP", local_ip="Any", direction="In"):
    return (f"Rule Name:                            {name}\n" + "-" * 70 + "\n"
            f"Enabled:                              {enabled}\nDirection:                            {direction}\n"
            f"Profiles:                             {profiles}\nLocalIP:                              {local_ip}\n"
            f"RemoteIP:                             {remote}\nProtocol:                             {protocol}\n"
            f"LocalPort:                            {port}\nRemotePort:                           Any\n"
            f"Program:                              {program}\nAction:                               {action}\n\n")


def fw(rules_text, profile_text=PROFILE_ON, ip="192.168.1.21", port=50201, programs=(r"C:\Python\python.exe",)):
    r = pf.Report()
    pf.check_local_firewall(r, "sec", "execution_listen_host", ip, port, list(programs), "192.168.1.10",
                            rules_text=rules_text, profile_text=profile_text)
    return r


def test_netsh_parsers():
    rules = pf.parse_netsh_rules(rule("A") + rule("B", program=r"C:\x\y.exe"))
    assert [x["name"] for x in rules] == ["A", "B"] and rules[1]["program"] == r"C:\x\y.exe"
    assert pf.parse_netsh_profiles(PROFILE_ON) == {"Private": True}
    assert pf.parse_netsh_profiles(PROFILE_OFF) == {"Private": False}


@pytest.mark.parametrize("spec, port, ok", [("Any", 5, True), ("50201", 50201, True), ("80,50201", 50201, True),
                                            ("50000-50300", 50201, True), ("50000-50100", 50201, False), ("RPC", 50201, False)])
def test_port_spec_matching(spec, port, ok):
    assert pf._port_matches(spec, port) is ok


def test_no_rule_fails_with_exact_command():
    c = one(fw(rule("other", port="80")), "sec", "execution_listen_host firewall")
    assert c.status == pf.FAIL and "no enabled inbound Allow rule matches TCP 50201" in c.detail
    assert "-LocalPort 50201 -RemoteAddress 192.168.1.10" in c.hint and "-Profile Private" in c.hint


def test_restricted_allow_passes():
    c = one(fw(rule("dllm", remote="192.168.1.10/255.255.255.255")), "sec", "execution_listen_host firewall")
    assert c.status == pf.PASS


def test_broad_allow_is_a_least_privilege_warning():
    c = one(fw(rule("open", remote="Any")), "sec", "execution_listen_host firewall")
    assert c.status == pf.WARN and "ANY remote address" in c.detail


def test_block_rule_wins():
    c = one(fw(rule("dllm", remote="192.168.1.10") + rule("python block", port="Any", action="Block",
                                                          program=r"C:\Python\python.exe")), "sec", "execution_listen_host firewall")
    assert c.status == pf.FAIL and "BLOCK" in c.detail and "python block" in c.detail


@pytest.mark.parametrize("kw", [dict(enabled="No"), dict(profiles="Public"), dict(protocol="UDP"), dict(direction="Out"),
                                dict(program=r"C:\other\app.exe"), dict(local_ip="192.168.1.99")])
def test_non_matching_rules_are_ignored(kw):
    c = one(fw(rule("x", remote="192.168.1.10", **kw)), "sec", "execution_listen_host firewall")
    assert c.status == pf.FAIL  # nothing applicable -> no allow rule


def test_firewall_off_is_a_warning():
    c = one(fw(rule("x"), PROFILE_OFF), "sec", "execution_listen_host firewall")
    assert c.status == pf.WARN and "OFF" in c.detail


def test_public_profile_program_rules_are_flagged():
    r = fw(rule("dllm", remote="192.168.1.10") + rule("Python", port="Any", profiles="Public", program=r"C:\Python\python.exe"))
    c = one(r, "sec", "execution_listen_host public-profile rules")
    assert c.status == pf.WARN and "'Python'" in c.detail


def test_unparseable_output_is_skipped_not_guessed():
    c = one(fw("Regelname: x\n", "Privates Profil\n"), "sec", "execution_listen_host firewall")
    assert c.status == pf.SKIP


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Firewall only")
def test_live_netsh_is_readable_here():
    """Against this machine's real firewall: the check must reach a verdict (not SKIP)."""
    r = pf.Report()
    pf.check_local_firewall(r, "sec", "probe", "192.168.255.254", free_port(), [sys.executable], "192.168.255.1")
    assert one(r, "sec", "probe firewall").status in (pf.PASS, pf.WARN, pf.FAIL)


# ---- orchestrator config check -------------------------------------------------------


def ocfg(tmp_path, text):
    p = tmp_path / "orchestrator.yaml"
    p.write_text(text)
    r = pf.Report()
    pf.check_orchestrator_config(r, str(p))
    return r


@pytest.mark.parametrize("text, name, status, fragment", [
    ('grpc_address: "8.8.8.8:9090"\n', "grpc_address", pf.FAIL, "public"),
    ('grpc_address: "my-pc:9090"\n', "grpc_address", pf.FAIL, "hostname"),
    ('grpc_address: "0.0.0.0:9090"\n', "grpc_address", pf.WARN, "every interface"),
    ('grpc_address: ":9090"\n', "grpc_address", pf.WARN, "every interface"),
    ('grpc_address: "127.0.0.1:9090"\n', "grpc_address", pf.WARN, "workers on OTHER machines cannot register"),
    ('rest_address: "127.0.0.1:{rest}"\n', "rest_address", pf.INFO, "loopback"),
    ('grpc_address: "10.255.255.254:9090"\n', "grpc_address ownership", pf.FAIL, "no interface with address 10.255.255.254"),
    ('grpc_address: "nonsense"\n', "grpc_address", pf.FAIL, "not HOST:PORT"),
    ('allow_simulated_workers: true\n', "allow_simulated_workers", pf.WARN, "SIMULATED"),
])
def test_orchestrator_config_checks(tmp_path, monkeypatch, text, name, status, fragment):
    monkeypatch.delenv("ORCHESTRATOR_GRPC_ADDRESS", raising=False)
    monkeypatch.delenv("ORCHESTRATOR_REST_ADDRESS", raising=False)
    c = one(ocfg(tmp_path, text.format(rest=free_port())), "orchestrator config", name)
    assert c.status == status and fragment in c.detail, (c.status, c.detail)


def test_orchestrator_env_override_wins_like_the_go_loader(tmp_path, monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_GRPC_ADDRESS", "8.8.8.8:9090")
    c = one(ocfg(tmp_path, 'grpc_address: "127.0.0.1:9090"\n'), "orchestrator config", "grpc_address")
    assert c.status == pf.FAIL and "8.8.8.8" in c.detail


def test_missing_orchestrator_config(tmp_path):
    r = pf.Report()
    pf.check_orchestrator_config(r, str(tmp_path / "nope.yaml"))
    assert one(r, "orchestrator config", "file").status == pf.FAIL


@needs_orch
@pytest.mark.parametrize("addr, starts", [("8.8.8.8:{p}", False), ("my-pc.example:{p}", False), ("127.0.0.1:{p}", True)])
def test_go_orchestrator_agrees_with_the_python_check(tmp_path, addr, starts):
    """Parity: what preflight calls FAIL, the real orchestrator.exe refuses to start with."""
    cfg = tmp_path / "o.yaml"
    cfg.write_text(f'grpc_address: "{addr.format(p=free_port())}"\nrest_address: "127.0.0.1:{free_port()}"\n')
    proc = subprocess.Popen([str(ORCH_EXE), "--config", str(cfg)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        out, _ = proc.communicate(timeout=4)
        exited = True
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate()
        exited = False
    assert exited is (not starts), out
    if not starts:
        assert proc.returncode != 0 and "grpc_address" in out


# ---- worker-side orchestrator probe -----------------------------------------------------


def test_probe_orchestrator_reports_startup_order(tmp_path):
    cfg = tmp_path / "w.yaml"
    cfg.write_text(f'worker_id: "w"\norchestrator_address: "127.0.0.1:{free_port()}"\n')
    r = pf.Report()
    pf.check_worker_config(r, str(cfg), probe_orchestrator=True)
    c = one(r, "worker config", "orchestrator_address reachable")
    assert c.status == pf.FAIL and "Startup order" in c.hint
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        cfg.write_text(f'worker_id: "w"\norchestrator_address: "127.0.0.1:{s.getsockname()[1]}"\n')
        r = pf.Report()
        pf.check_worker_config(r, str(cfg), probe_orchestrator=True)
    assert one(r, "worker config", "orchestrator_address reachable").status == pf.PASS


def test_probe_is_opt_in(tmp_path):
    cfg = tmp_path / "w.yaml"
    cfg.write_text('worker_id: "w"\n')
    r = pf.Report()
    pf.check_worker_config(r, str(cfg))
    assert not [c for c in r.checks if c.name == "orchestrator_address reachable"]


# ---- config generators -------------------------------------------------------------------


def gen(*argv):
    return P8.main(list(argv))


def test_worker_config_generator_round_trips(tmp_path, monkeypatch):
    from worker.config.loader import load

    for k in ("WORKER_ID", "WORKER_LISTEN_HOST", "WORKER_EXECUTION_LISTEN_HOST", "WORKER_EXECUTION_ENABLED"):
        monkeypatch.delenv(k, raising=False)
    out = tmp_path / "worker-b.yaml"
    assert gen("worker-config", "--worker-id", "computer-2-b", "--orchestrator", "192.168.1.10:9090", "--execution-host",
               "192.168.1.21", "--execution-port", "50201", "--out", str(out)) == 0
    cfg = load(str(out), env_file=None)
    assert (cfg.worker_id, cfg.orchestrator_address, cfg.execution_listen_host, cfg.execution_listen_port) == (
        "computer-2-b", "192.168.1.10:9090", "192.168.1.21", 50201)
    assert cfg.execution_enabled and cfg.listen_host == "127.0.0.1" and cfg.simulated_profile == ""
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        gen("worker-config", "--worker-id", "x", "--orchestrator", "192.168.1.10:9090", "--execution-host", "192.168.1.21",
            "--out", str(out))


@pytest.mark.parametrize("argv, fragment", [
    (["--execution-host", "8.8.8.8"], "PUBLIC"),
    (["--execution-host", "0.0.0.0"], "every interface"),
    (["--execution-host", "my-pc"], "hostname"),
    (["--execution-host", "127.0.0.1"], "--allow-loopback"),
    (["--execution-host", "192.168.1.21", "--worker-id", "bad id!"], "--worker-id"),
    (["--execution-host", "192.168.1.21", "--execution-port", "50100"], "must differ"),
    (["--execution-host", "192.168.1.21", "--orchestrator", "192.168.1.10"], "HOST:PORT"),
])
def test_worker_config_generator_refuses_unsafe_values(tmp_path, argv, fragment):
    base = {"--worker-id": "w", "--orchestrator": "192.168.1.10:9090"}
    for k, v in zip(argv[::2], argv[1::2]):
        base[k] = v
    args = ["worker-config", "--out", str(tmp_path / "w.yaml")] + [x for kv in base.items() for x in kv]
    with pytest.raises(SystemExit, match=fragment.replace("-", r"\-") if fragment.startswith("--") else fragment):
        gen(*args)
    assert not (tmp_path / "w.yaml").exists()


def test_orchestrator_config_generator(tmp_path, monkeypatch):
    monkeypatch.delenv("ORCHESTRATOR_GRPC_ADDRESS", raising=False)
    out = tmp_path / "o.yaml"
    with pytest.raises(SystemExit, match="PUBLIC"):
        gen("orchestrator-config", "--lan-ip", "8.8.8.8", "--out", str(out))
    with pytest.raises(SystemExit, match="--allow-loopback"):
        gen("orchestrator-config", "--lan-ip", "127.0.0.1", "--out", str(out))
    assert gen("orchestrator-config", "--lan-ip", "127.0.0.1", "--allow-loopback", "--grpc-port", str(free_port()),
               "--rest-port", str(free_port()), "--ram-margin-percent", "2", "--out", str(out)) == 0
    text = out.read_text()
    assert "DEVIATION" in text and "ram_safety_margin_percent: 2" in text and "allow_simulated_workers: false" in text
    r = pf.Report()
    pf.check_orchestrator_config(r, str(out))
    assert not r.failed, [(c.name, c.detail) for c in r.checks if c.status == pf.FAIL]


@needs_orch
def test_generated_orchestrator_config_starts_the_real_orchestrator(tmp_path):
    import time
    import urllib.request

    rest = free_port()
    out = tmp_path / "o.yaml"
    gen("orchestrator-config", "--lan-ip", "127.0.0.1", "--allow-loopback", "--grpc-port", str(free_port()),
        "--rest-port", str(rest), "--out", str(out))
    proc = subprocess.Popen([str(ORCH_EXE), "--config", str(out)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(40):
            try:
                body = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{rest}/health", timeout=1).read())
                break
            except OSError:
                time.sleep(0.25)
        else:
            pytest.fail("orchestrator did not answer")
        assert body["status"] == "ok"
    finally:
        proc.kill()
        proc.wait(timeout=10)


def test_run_needs_two_endpoints(tmp_path):
    with pytest.raises(SystemExit, match="at least two"):
        gen("run", "--endpoint", "a=127.0.0.1:1", "--out", str(tmp_path / "r"))


# ---- the multi-machine verdict (synthetic results: SIMULATED) ---------------------------


def result(*stages):
    return {"stages": [{"worker_id": w, "endpoint": ep, "worker_reported": {"hostname": h, "pid": pid}}
                       for w, ep, h, pid in stages]}


def view(**peers):
    return {w: {"observed_peer_ip": ip, "hostname": "?", "address": "127.0.0.1:50100"} for w, ip in peers.items()}


def test_judge_two_machines():
    r = result(("a", "127.0.0.1:50200", "PC-1", 10), ("b", "192.168.1.21:50201", "PC-2", 20))
    status, detail, ev = P8.judge(r, view(a="192.168.1.10", b="192.168.1.21"))
    assert status == "PASS" and ev["classification"] == "REAL_MULTI_MACHINE"


def test_judge_rejects_renamed_hosts_on_one_machine():
    """Different self-reported hostnames but every registration from loopback: not multi-machine."""
    r = result(("a", "127.0.0.1:1", "PC-1", 10), ("b", "127.0.0.1:2", "PC-2", 20))
    status, detail, ev = P8.judge(r, view(a="127.0.0.1", b="127.0.0.1"))
    assert status == "FAIL" and ev["classification"] == "REAL_MULTI_PROCESS" and "NOT met" in detail


def test_judge_rejects_one_multihomed_machine():
    """Two LAN IPs but one reported hostname: not multi-machine."""
    r = result(("a", "192.168.1.10:1", "PC-1", 10), ("b", "10.0.0.5:2", "PC-1", 20))
    assert P8.judge(r, view(a="192.168.1.10", b="10.0.0.5"))[2]["classification"] == "REAL_MULTI_PROCESS"


def test_judge_rehearsal_flag_never_upgrades_the_class():
    r = result(("a", "127.0.0.1:1", "PC-1", 10), ("b", "127.0.0.1:2", "PC-1", 20))
    status, detail, ev = P8.judge(r, view(a="127.0.0.1", b="127.0.0.1"), allow_single_host=True)
    assert status == "PASS" and ev["classification"] == "REAL_MULTI_PROCESS" and "NOT claimed" in detail


# ---- full loopback rehearsals (REAL_MULTI_PROCESS) --------------------------------------


@needs_orch
def test_rehearsal_end_to_end(tmp_path):
    rc = P8.rehearse(tmp_path / "reh", repeat=2, allow_single_host=True, with_operator=True)
    report = json.loads((tmp_path / "reh" / "report.json").read_text())
    steps = {s["id"]: s for s in report["steps"]}
    assert rc == 0 and report["result"] == "PASS", [(s["id"], s["detail"]) for s in report["steps"] if s["status"] != "PASS"]
    assert report["classification"] == "REAL_MULTI_PROCESS"  # one machine: never REAL_MULTI_MACHINE
    assert [steps[f"P{i}"]["status"] for i in range(2, 9)] == ["PASS"] * 7
    ev = steps["P7"]["evidence"]
    assert ev["distinct_non_loopback_peer_ips"] == [] and len(ev["distinct_reported_hostnames"]) == 1
    assert all(v["orchestrator_observed_peer_ip"] == "127.0.0.1" for v in ev["per_worker"].values())
    assert "executed=False" in steps["P8"]["detail"]
    md = (tmp_path / "reh" / "report.md").read_text()
    assert "REAL_MULTI_PROCESS" in md and "REAL_MULTI_MACHINE`" not in md


@needs_orch
def test_strict_rehearsal_fails_the_multi_machine_criterion(tmp_path):
    rc = P8.rehearse(tmp_path / "strict", repeat=1, allow_single_host=False, with_operator=False)
    report = json.loads((tmp_path / "strict" / "report.json").read_text())
    steps = {s["id"]: s for s in report["steps"]}
    assert rc == 1 and report["result"] == "FAIL"
    assert steps["P7"]["status"] == "FAIL" and "NOT met" in steps["P7"]["detail"]
    assert steps["P5"]["status"] == "FAIL"  # preflight --require-multi-host also refuses
    assert steps["P6"]["status"] == "PASS" and steps["P8"]["status"] == "SKIP"
    assert report["classification"] == "REAL_MULTI_PROCESS"


@needs_orch
def test_rehearsal_report_contains_no_lan_addresses(tmp_path):
    """Rehearsal reports may be committed: no private/public address of this machine may appear in them."""
    import ipaddress

    P8.rehearse(tmp_path / "r", repeat=1, allow_single_host=True, with_operator=False)
    mine = [a for _, a, k in pf.local_ipv4() if k != "loopback"]
    files = [p for p in (tmp_path / "r").rglob("*") if p.is_file()]
    assert len(files) >= 8
    for path in files:  # every artifact: reports, preflight output, plans, results, process logs, configs
        text = path.read_text(encoding="utf-8", errors="replace")
        leaked = [a for a in mine if a in text]
        assert not leaked, (str(path), leaked)
    assert ipaddress.ip_address("127.0.0.1")  # loopback evidence is kept
