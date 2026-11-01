"""Preflight diagnostics for a LAN deployment (Phase 7).

    python -m partitioner preflight [--worker-config worker.yaml ...]
        [--orchestrator http://IP:8080 [--expect-worker ID ...]]
        [--endpoint ID=HOST:PORT ...] [--plan plan.json [--require-multi-host]]

Sections run only when their inputs are given; the runtime and local
network sections always run. Every check has a status:

  PASS  tested, and fine
  WARN  tested, works, but needs attention
  FAIL  tested, and broken (each FAIL carries a concrete hint)
  SKIP  NOT tested (missing input, or an earlier check failed)
  INFO  facts, no verdict

A check never reports a remote endpoint as reachable unless it actually
connected to it: each check says what it exercised (`tested`).
Exit code: 1 if any check FAILs, else 0.
"""

from __future__ import annotations

import dataclasses
import errno
import json
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from importlib import metadata
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PASS, WARN, FAIL, SKIP, INFO = "PASS", "WARN", "FAIL", "SKIP", "INFO"
REPO = Path(__file__).resolve().parents[2]

HINTS = {
    "dns": ("Use the machine's private IPv4 address instead of a hostname (name resolution between Windows machines "
            "on a home LAN is unreliable), or fix DNS / the hosts file."),
    "refused": ("The host answered, but nothing is listening on that port. Is the process running? Is the port right? "
                "Does it listen on the LAN IP? A service bound to 127.0.0.1 refuses connections from other machines."),
    "timeout": ("No answer before the timeout. This is usually Windows Firewall dropping inbound TCP on the target "
                "(add the LocalSubnet rule from docs/DEPLOYMENT_LAN.md and set the network profile to Private), a "
                "wrong IP, or the machine is off or asleep."),
    "unreachable": ("No route to the host. Are both machines on the same network and subnet? Compare `ipconfig` on "
                    "both; Wi-Fi guest networks often isolate clients."),
    "error": "Unexpected socket error; see the detail.",
    # Windows retries a refused loopback connection for ~2 s, so a closed
    # local port can surface as a timeout rather than as "refused".
    "timeout-loopback": ("Nothing answered on this machine's own port. Most likely nothing is listening there (on "
                         "Windows a closed loopback port can look like a timeout rather than a refusal). Is the "
                         "process running, and is the port right?"),
}


def hint_for(probe: "Probe") -> str:
    if probe.kind == "timeout" and probe.ip and _classify(probe.ip) == "loopback":
        return HINTS["timeout-loopback"]
    return HINTS[probe.kind]
_UNREACHABLE = {errno.EHOSTUNREACH, errno.ENETUNREACH, 10065, 10051}  # 1006x = Winsock


@dataclasses.dataclass
class Check:
    section: str
    name: str
    status: str
    detail: str
    hint: str = ""
    tested: str = ""
    data: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class Report:
    checks: List[Check] = dataclasses.field(default_factory=list)

    def add(self, section, name, status, detail, hint="", tested="", **data) -> Check:
        c = Check(section, name, status, detail, hint, tested, data)
        self.checks.append(c)
        return c

    @property
    def failed(self) -> bool:
        return any(c.status == FAIL for c in self.checks)

    def counts(self) -> Dict[str, int]:
        out = {s: 0 for s in (PASS, WARN, FAIL, SKIP, INFO)}
        for c in self.checks:
            out[c.status] += 1
        return out

    def to_dict(self) -> dict:
        return {"preflight_schema_version": 1, "hostname": socket.gethostname(), "counts": self.counts(),
                "ok": not self.failed, "checks": [dataclasses.asdict(c) for c in self.checks]}

    def render(self) -> str:
        lines, section = [], None
        for c in self.checks:
            if c.section != section:
                section = c.section
                lines.append(f"\n== {section} ==")
            lines.append(f"[{c.status}] {c.name}: {c.detail}")
            if c.tested:
                lines.append(f"       tested: {c.tested}")
            if c.hint and c.status in (FAIL, WARN, SKIP):
                lines.append(f"       hint: {c.hint}")
        counts = self.counts()
        lines.append("\nSUMMARY: " + ", ".join(f"{v} {k}" for k, v in counts.items())
                     + (" -- NOT READY (fix every FAIL)" if self.failed else " -- no failures"))
        return "\n".join(lines)


# ---------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------


@dataclasses.dataclass
class Probe:
    ok: bool
    kind: str  # "ok" | dns | refused | timeout | unreachable | error
    detail: str
    ip: Optional[str] = None
    seconds: float = 0.0


def tcp_probe(host: str, port: int, timeout: float) -> Probe:
    """DNS resolution, then a real TCP connect (no data sent)."""
    t = time.perf_counter()
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        return Probe(False, "dns", f"cannot resolve {host!r}: {exc}", seconds=time.perf_counter() - t)
    infos.sort(key=lambda i: i[0] != socket.AF_INET)  # prefer IPv4
    family, _, _, _, sockaddr = infos[0]
    ip = sockaddr[0]
    try:
        with socket.socket(family, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect(sockaddr)
        return Probe(True, "ok", f"TCP connect to {ip}:{port} succeeded", ip, time.perf_counter() - t)
    except ConnectionRefusedError:
        return Probe(False, "refused", f"{ip}:{port} refused the connection", ip, time.perf_counter() - t)
    except (socket.timeout, TimeoutError):
        return Probe(False, "timeout", f"no answer from {ip}:{port} within {timeout}s", ip, time.perf_counter() - t)
    except OSError as exc:
        kind = "unreachable" if (exc.errno in _UNREACHABLE or getattr(exc, "winerror", None) in _UNREACHABLE) else "error"
        return Probe(False, kind, f"{ip}:{port}: {exc}", ip, time.perf_counter() - t)


def split_host_port(addr: str) -> Tuple[str, int]:
    host, sep, port = addr.rpartition(":")
    if not sep or not host or not port.isdigit() or not 0 < int(port) < 65536:
        raise ValueError(f"{addr!r} is not HOST:PORT")
    return host.strip("[]"), int(port)


def _classify(host: str) -> str:
    from partitioner.workers import _ensure_worker_importable

    _ensure_worker_importable()
    from worker.config.bind_policy import classify

    return classify(host)


def local_ipv4() -> List[Tuple[str, str, str]]:
    """(interface, address, class) for every IPv4 address of this machine."""
    import psutil

    out = []
    for iface, addrs in psutil.net_if_addrs().items():
        for a in addrs:
            if a.family == socket.AF_INET:
                kind = _classify(a.address)
                if a.address.startswith("169.254."):
                    kind = "link-local (APIPA: no DHCP lease; not a usable LAN address)"
                out.append((iface, a.address, kind))
    return sorted(out, key=lambda x: (x[2] != "private", x[0]))


# ---------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------


def _pins() -> Dict[str, str]:
    pins = {}
    for req in (REPO / "worker" / "requirements.txt", REPO / "model-partitioner" / "requirements.txt"):
        if not req.exists():
            continue
        for line in req.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if "==" in line and not line.startswith("-"):
                name, ver = line.split("==", 1)
                pins[name.strip()] = ver.strip()
    return pins


def check_runtime(r: Report) -> None:
    sec = "runtime"
    v = sys.version_info
    status = PASS if (v.major, v.minor) == (3, 13) else WARN
    r.add(sec, "python", status, f"Python {sys.version.split()[0]} ({sys.executable})",
          "Computer 1 runs 3.13.3; use the same minor version on every machine to avoid wheel and behaviour "
          "differences." if status == WARN else "", tested="this interpreter")
    for name, pinned in sorted(_pins().items()):
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            r.add(sec, name, FAIL, f"not installed (pinned {pinned})",
                  f"pip install -r worker/requirements.txt -r model-partitioner/requirements.txt", tested="importlib.metadata")
            continue
        same = installed.split("+")[0] == pinned.split("+")[0]
        r.add(sec, name, PASS if same else WARN, f"{installed} (pinned {pinned})",
              "" if same else "Version differs from the pin verified on Computer 1; usually fine, but note it if "
                              "results differ between machines.", tested="importlib.metadata")
    try:
        import numpy  # noqa: F401 -- needed by the tensor codec

        r.add(sec, "numpy", PASS, f"{numpy.__version__} (needed by the raw-le-c-v1 tensor codec)", tested="import")
    except ImportError:
        r.add(sec, "numpy", FAIL, "not importable", "pip install numpy", tested="import")
    try:
        from partitioner.execution import wire

        pb = wire.pb2()
        has_info = hasattr(pb, "GetInfoRequest")
        r.add(sec, "protocol stubs", PASS if has_info else FAIL,
              f"worker.generated.stage_execution_pb2 imports; local protocol version {wire.PROTOCOL_VERSION}"
              + ("" if has_info else "; GetInfo missing (stubs older than Phase 7)"),
              "" if has_info else "Regenerate stubs: powershell -ExecutionPolicy Bypass -File scripts/generate_protos.ps1",
              tested="import")
    except Exception as exc:
        r.add(sec, "protocol stubs", FAIL, f"cannot import the StageExecution stubs: {exc}",
              "Run from the repository checkout (model-partitioner/ next to worker/).", tested="import")


def check_local_network(r: Report) -> None:
    sec = "this machine"
    r.add(sec, "hostname", INFO, socket.gethostname(), tested="socket.gethostname()")
    try:
        addrs = local_ipv4()
    except Exception as exc:
        r.add(sec, "IPv4 addresses", SKIP, f"could not list interfaces: {exc}")
        return
    for iface, addr, kind in addrs:
        r.add(sec, f"IPv4 {addr}", INFO, f"{kind} ({iface})", tested="psutil.net_if_addrs()")
    public = [a for _, a, k in addrs if k == "public"]
    if public:
        r.add(sec, "public interface", WARN, f"this machine has public IPv4 address(es) {public}",
              "Never bind any service to 0.0.0.0 on this machine: it would listen on the public address too. Use the "
              "private LAN IP explicitly.", tested="interface classification")
    else:
        r.add(sec, "public interface", PASS, "no public IPv4 address on any interface", tested="interface classification")


def port_owner(port: int) -> Optional[Tuple[int, str]]:
    """(pid, process name) of whatever LISTENs on this TCP port, if visible."""
    try:
        import psutil

        for c in psutil.net_connections(kind="tcp"):
            if c.status == psutil.CONN_LISTEN and c.laddr and c.laddr.port == port and c.pid:
                try:
                    return c.pid, psutil.Process(c.pid).name()
                except psutil.Error:
                    return c.pid, "?"
    except Exception:
        return None
    return None


def check_port_free(r: Report, sec: str, setting: str, host: str, port: int, ours: Sequence[str]) -> None:
    """Port free? If not: who holds it -- one of our own programs (it may
    simply be running already) or something unrelated (a real conflict)."""
    free, err = _can_bind(host, port)
    if free:
        r.add(sec, f"{setting} port {port}", PASS, "free", tested=f"bind({host}, {port})")
        return
    owner = port_owner(port)
    tested = f"bind({host}, {port}) + listening-socket owner (psutil)"
    if owner is None:
        r.add(sec, f"{setting} port {port}", WARN, f"in use ({err}); owner not visible",
              "Something already holds this port (maybe our own service). Check: netstat -ano | findstr :" + str(port)
              + "; reserved ranges: netsh int ipv4 show excludedportrange protocol=tcp", tested=tested)
    elif owner[1].lower() in {o.lower() for o in ours}:
        r.add(sec, f"{setting} port {port}", INFO if owner[1].lower() != "python.exe" else WARN,
              f"held by {owner[1]} (pid {owner[0]}): probably this service already running",
              "If it is not, stop pid " + str(owner[0]) + " or pick another port.", tested=tested)
    else:
        r.add(sec, f"{setting} port {port}", FAIL, f"held by an unrelated program: {owner[1]} (pid {owner[0]})",
              "Pick another port for this setting (and use it consistently in configs, firewall rules and --endpoint), "
              "or stop that program if you own it.", tested=tested)


def _can_bind(host: str, port: int) -> Tuple[bool, str]:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True, ""
        except OSError as exc:
            return False, str(exc)


# ---------------------------------------------------------------------
# Local Windows Firewall (Phase 8)
# ---------------------------------------------------------------------
#
# Read-only and heuristic: it reads `netsh advfirewall` output (0.4 s on
# Computer 1; the PowerShell cmdlets took 65 s) and predicts whether an
# inbound connection to a LAN-bound port would be admitted. The
# definitive test is a TCP connect FROM THE PEER machine (preflight
# --endpoint / --probe-orchestrator run there). netsh output is
# localized: on a non-English Windows the parser finds no rules and the
# check is SKIPPED, never guessed.

ORCHESTRATOR_EXE = REPO / "orchestrator" / "orchestrator.exe"


def parse_netsh_rules(text: str) -> List[dict]:
    rules: List[dict] = []
    cur: Optional[dict] = None
    for line in text.splitlines():
        if line.startswith("Rule Name:"):
            cur = {"name": line.split(":", 1)[1].strip()}
            rules.append(cur)
        elif cur is not None and ":" in line and not line.startswith("-"):
            key, value = line.split(":", 1)
            cur[key.strip().lower()] = value.strip()
    return rules


def parse_netsh_profiles(text: str) -> Dict[str, bool]:
    """Active profile name -> firewall on?"""
    out, current = {}, None
    for line in text.splitlines():
        if line.strip().endswith("Profile Settings:"):
            current = line.split()[0]
            out[current] = True
        elif current and line.startswith("State"):
            out[current] = line.split()[-1].upper() == "ON"
    return out


def _port_matches(spec: str, port: int) -> bool:
    for part in spec.replace(" ", "").split(","):
        if part.lower() == "any":
            return True
        if "-" in part:
            lo, _, hi = part.partition("-")
            if lo.isdigit() and hi.isdigit() and int(lo) <= port <= int(hi):
                return True
        elif part.isdigit() and int(part) == port:
            return True
    return False


def _norm(path: str) -> str:
    return str(Path(path)).lower() if path else ""


def matching_rules(rules: Sequence[dict], profiles: Sequence[str], ip: str, port: int, programs: Sequence[str]) -> List[dict]:
    progs = {_norm(p) for p in programs}
    out = []
    for rule in rules:
        if rule.get("enabled", "").lower() != "yes" or rule.get("direction", "").lower() != "in":
            continue
        if rule.get("protocol", "").upper() not in ("TCP", "ANY"):
            continue
        rprof = rule.get("profiles", "")
        if rprof.lower() != "any" and not any(p.lower() in rprof.lower() for p in profiles):
            continue
        if not _port_matches(rule.get("localport", ""), port):
            continue
        local_ip = rule.get("localip", "Any")
        if local_ip.lower() != "any" and ip not in local_ip:
            continue
        program = rule.get("program", "Any")
        if program.lower() != "any" and _norm(program) not in progs:
            continue
        out.append(rule)
    return out


def _netsh(*args: str) -> Optional[str]:
    import subprocess

    try:
        return subprocess.run(["netsh", "advfirewall", *args], capture_output=True, text=True, timeout=20,
                              errors="replace").stdout
    except (OSError, subprocess.SubprocessError):
        return None


def check_local_firewall(r: Report, sec: str, setting: str, ip: str, port: int, programs: Sequence[str],
                         peer_hint: str, rules_text: Optional[str] = None, profile_text: Optional[str] = None) -> None:
    """Can a peer on the LAN reach ip:port on THIS machine? (heuristic)"""
    name = f"{setting} firewall"
    if rules_text is None:
        if sys.platform != "win32":
            r.add(sec, name, SKIP, "not tested: only Windows Firewall is inspected")
            return
        rules_text = _netsh("firewall", "show", "rule", "name=all", "dir=in", "verbose")
        profile_text = _netsh("show", "currentprofile")
    rules = parse_netsh_rules(rules_text or "")
    profiles = parse_netsh_profiles(profile_text or "")
    if not rules or not profiles:
        r.add(sec, name, SKIP, "not tested: could not read or parse `netsh advfirewall` output (non-English Windows?)",
              "Check the rules by hand: DEPLOYMENT_LAN.md, Step 3.")
        return
    tested = f"netsh advfirewall (read-only), active profile(s) {sorted(profiles)}; a TCP connect from the peer is the real test"
    off = [p for p, on in profiles.items() if not on]
    if off:
        r.add(sec, name, WARN, f"Windows Firewall is OFF for the active profile(s) {off}: {ip}:{port} is open to the whole "
              "network", "Turn the firewall back on and add the least-privilege rule from DEPLOYMENT_LAN.md.", tested=tested)
        return
    found = matching_rules(rules, list(profiles), ip, port, programs)
    blocks = [x for x in found if x.get("action", "").lower() == "block"]
    allows = [x for x in found if x.get("action", "").lower() == "allow"]
    rule_cmd = (f'New-NetFirewallRule -DisplayName "dllm {setting} {port}" -Direction Inbound -Protocol TCP -LocalPort {port} '
                f"-RemoteAddress {peer_hint} -Action Allow -Profile Private")
    if blocks:
        r.add(sec, name, FAIL, f"enabled inbound BLOCK rule(s) {sorted({b['name'] for b in blocks})} match TCP {port}; "
              "block rules win over allow rules", "Remove or disable that rule (often created by clicking 'Cancel' on a "
              "Windows Firewall prompt): Get-NetFirewallRule -DisplayName '<name>' | Remove-NetFirewallRule", tested=tested)
    elif not allows:
        r.add(sec, name, FAIL, f"no enabled inbound Allow rule matches TCP {port} on the {sorted(profiles)} profile: "
              "connections from the peer will be dropped (they show up there as a timeout)",
              f"In an elevated PowerShell: {rule_cmd}", tested=tested)
    else:
        broad = [a["name"] for a in allows if a.get("remoteip", "Any").lower() == "any"]
        detail = "; ".join(f"{a['name']!r} (remote {a.get('remoteip')}, port {a.get('localport')}, program "
                           f"{a.get('program')})" for a in allows)
        if broad:
            r.add(sec, name, WARN, f"allowed by {detail}, but {sorted(set(broad))} admit ANY remote address",
                  f"Least privilege: restrict to the peer, e.g. {rule_cmd}, and remove the broad rule.", tested=tested)
        else:
            r.add(sec, name, PASS, f"allowed only from specific remote addresses: {detail}", tested=tested)
    public = [x for x in rules if x.get("enabled", "").lower() == "yes" and x.get("direction", "").lower() == "in"
              and x.get("action", "").lower() == "allow" and "public" in x.get("profiles", "").lower()
              and x.get("program", "Any").lower() != "any" and _norm(x["program"]) in {_norm(p) for p in programs}
              and x.get("localport", "").lower() == "any"]
    if public:
        r.add(sec, f"{setting} public-profile rules", WARN,
              f"{len(public)} enabled inbound Allow rule(s) {sorted({x['name'] for x in public})} admit this program on ANY "
              "port on the Public profile", "On a public network these would expose any LAN-bound listener of this program. "
              "Review: Get-NetFirewallRule -DisplayName '<name>' | Format-List Profile,Action", tested=tested)


def check_orchestrator_config(r: Report, path: str, peer_hint: str = "<COMPUTER-2-IP>") -> None:
    """Computer 1: the orchestrator's own config, read the way the Go
    orchestrator reads it (file, then ORCHESTRATOR_* environment)."""
    import os

    import yaml

    sec = f"orchestrator config {path}"
    if not Path(path).exists():
        r.add(sec, "file", FAIL, "file not found; the orchestrator would run on defaults (loopback only)",
              "Create it: python scripts/phase8_lan.py orchestrator-config --lan-ip <COMPUTER-1-IP> --out ...", tested="filesystem")
        return
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError("top level must be a mapping")
    except Exception as exc:
        r.add(sec, "parse", FAIL, f"{type(exc).__name__}: {exc}", "Fix the YAML.", tested="yaml.safe_load")
        return
    r.add(sec, "parse", PASS, "parsed; ORCHESTRATOR_REST_ADDRESS / ORCHESTRATOR_GRPC_ADDRESS environment overrides applied "
          "as the orchestrator would", tested="yaml.safe_load")
    listeners = {
        "grpc_address": os.environ.get("ORCHESTRATOR_GRPC_ADDRESS") or str(data.get("grpc_address", "127.0.0.1:9090")),
        "rest_address": os.environ.get("ORCHESTRATOR_REST_ADDRESS") or str(data.get("rest_address", "127.0.0.1:8080")),
    }
    for setting, addr in listeners.items():
        host, sep, port = addr.rpartition(":")
        host = host.strip("[]") or "0.0.0.0"
        if not sep or not port.isdigit():
            r.add(sec, setting, FAIL, f"{addr!r} is not HOST:PORT", tested="config value")
            continue
        port_n, kind = int(port), _classify(host)
        if kind in ("public", "hostname"):
            r.add(sec, setting, FAIL, f"{addr}: {kind} -- the orchestrator refuses to start (config.CheckBindAddress)",
                  "Use 127.0.0.1 or this machine's private LAN IP.", tested="bind policy (mirror of the Go check)")
            continue
        if kind == "all-interfaces":
            r.add(sec, setting, WARN, f"{addr} listens on every interface", "Use the private LAN IP explicitly.",
                  tested="bind policy")
        elif kind == "loopback":
            what = ("workers on OTHER machines cannot register" if setting == "grpc_address"
                    else "fine: the coordinator and preflight run on this machine")
            r.add(sec, setting, INFO if setting == "rest_address" else WARN, f"{addr} is loopback: {what}",
                  "" if setting == "rest_address" else "For a LAN test set grpc_address to this machine's private LAN IP.",
                  tested="bind policy")
        else:
            r.add(sec, setting, PASS if setting == "grpc_address" else WARN, f"{addr} ({kind})",
                  "" if setting == "grpc_address" else "Least privilege: the REST API is only needed on this machine; "
                  "keep rest_address on 127.0.0.1 unless another machine must read it.", tested="bind policy")
        if kind in ("loopback", "private") and host != "localhost":
            ok, err = _can_bind(host, 0)
            if not ok:
                r.add(sec, f"{setting} ownership", FAIL, f"this machine has no interface with address {host}: {err}",
                      "Run ipconfig and use the current IPv4 address (DHCP may have changed it).", tested=f"bind({host}, 0)")
                continue
            check_port_free(r, sec, setting, host, port_n, ["orchestrator.exe", "orchestrator"])
        if kind in ("private", "all-interfaces") and setting == "grpc_address":
            check_local_firewall(r, sec, setting, host, port_n, [str(ORCHESTRATOR_EXE)], peer_hint)
    if data.get("allow_simulated_workers"):
        r.add(sec, "allow_simulated_workers", WARN, "true: SIMULATED profiles would be accepted",
              "Set allow_simulated_workers: false for a real deployment.", tested="config value")
    else:
        r.add(sec, "allow_simulated_workers", PASS, "false", tested="config value")


def check_worker_config(r: Report, path: str, probe_orchestrator: bool = False,
                        peer_hint: str = "<COORDINATOR-IP>") -> Optional[object]:
    from partitioner.workers import _ensure_worker_importable

    _ensure_worker_importable()
    from worker.config.bind_policy import BindAddressError, check_bind_host
    from worker.config.loader import load

    sec = f"worker config {path}"
    if not Path(path).exists():
        r.add(sec, "file", FAIL, "file not found; the worker would silently run on defaults",
              "Copy worker/worker.example.yaml and edit it.", tested="filesystem")
        return None
    try:
        cfg = load(path)
    except Exception as exc:
        r.add(sec, "parse", FAIL, f"{type(exc).__name__}: {exc}", "Fix the YAML (a key: value per line).", tested="worker loader")
        return None
    r.add(sec, "parse", PASS, "loaded with the worker's own loader (WORKER_* environment variables and .env applied)",
          tested="worker.config.loader.load")
    if cfg.worker_id.startswith(socket.gethostname() + "-") and cfg.worker_id.split("-")[-1].isdigit():
        r.add(sec, "worker_id", FAIL, f"{cfg.worker_id!r} is auto-generated (hostname-pid) and changes on every start",
              "Set a stable worker_id, e.g. worker_id: \"computer-2\".", tested="config value")
    else:
        r.add(sec, "worker_id", PASS, repr(cfg.worker_id), tested="config value")

    listeners = [("listen_host", cfg.listen_host, cfg.listen_port, True, "")]
    if cfg.execution_enabled:
        listeners.append(("execution_listen_host", cfg.execution_listen_host, cfg.execution_listen_port,
                          cfg.execution_allow_all_interfaces, "execution_allow_all_interfaces"))
    for setting, host, port, allow_all, flag in listeners:
        kind = _classify(host)
        try:
            check_bind_host(host, allow_all, setting=setting, allow_flag=flag)
        except BindAddressError as exc:
            r.add(sec, setting, FAIL, str(exc), "Use 127.0.0.1 (this machine only) or this machine's private LAN IP.",
                  tested="bind policy")
            continue
        if kind == "all-interfaces":
            r.add(sec, setting, WARN, f"{host}:{port} listens on every interface", "Prefer the private LAN IP.",
                  tested="bind policy")
        elif kind == "loopback":
            r.add(sec, setting, INFO, f"{host}:{port} is loopback: reachable from THIS machine only",
                  tested="bind policy")
        else:
            r.add(sec, setting, PASS, f"{host}:{port} ({kind})", tested="bind policy")
        if kind in ("loopback", "private") and host != "localhost":
            ok, err = _can_bind(host, 0)
            if not ok:
                r.add(sec, f"{setting} ownership", FAIL, f"this machine has no interface with address {host}: {err}",
                      "Run `ipconfig` and use this machine's current IPv4 address (DHCP may have changed it).",
                      tested=f"bind({host}, 0)")
                continue
            r.add(sec, f"{setting} ownership", PASS, f"{host} belongs to this machine", tested=f"bind({host}, 0)")
            check_port_free(r, sec, setting, host, port, ["python.exe", "python"])
        if kind in ("private", "all-interfaces") and setting == "execution_listen_host":
            check_local_firewall(r, sec, setting, host, port, [sys.executable], peer_hint)
    if cfg.execution_enabled and cfg.listen_port == cfg.execution_listen_port:
        r.add(sec, "ports", FAIL, f"listen_port and execution_listen_port are both {cfg.listen_port}",
              "Use two different ports, e.g. 50100 (Health) and 50200 (StageExecution).", tested="config values")
    if not cfg.execution_enabled:
        r.add(sec, "execution_enabled", WARN, "false: this worker cannot run pipeline stages",
              "Set execution_enabled: true on every worker that should execute stages.", tested="config value")
    elif cfg.simulated_profile:
        r.add(sec, "execution_enabled", FAIL, "a SIMULATED worker refuses to start StageExecution",
              "Remove simulated_profile for a real deployment.", tested="config values")
    else:
        r.add(sec, "execution_enabled", PASS, "true", tested="config value")
    if cfg.execution_max_message_mib * 2**20 < cfg.execution_max_tensor_mib * 2**20 + 256 * 1024:
        r.add(sec, "message limit", FAIL, "execution_max_message_mib leaves no room for the plan beside a full tensor",
              "Keep execution_max_message_mib at least execution_max_tensor_mib + 1.", tested="config values")
    try:
        ohost, oport = split_host_port(cfg.orchestrator_address)
        okind = _classify(ohost)
        lan = any(_classify(h) == "private" for _, h, _, _, _ in listeners)
        if okind == "loopback" and lan:
            r.add(sec, "orchestrator_address", WARN, f"{cfg.orchestrator_address} is loopback, but this worker listens on "
                  "the LAN", "On Computer 2, set orchestrator_address to Computer 1's private IP, e.g. 192.168.1.10:9090.",
                  tested="config value")
        else:
            r.add(sec, "orchestrator_address", PASS, f"{cfg.orchestrator_address} ({okind})", tested="config value")
        if probe_orchestrator:
            probe = tcp_probe(ohost, oport, 3.0)
            r.add(sec, "orchestrator_address reachable", PASS if probe.ok else FAIL, probe.detail,
                  "" if probe.ok else hint_for(probe) + " Startup order: the orchestrator must already be running.",
                  tested=f"DNS + TCP connect {ohost}:{oport} from this machine")
    except ValueError as exc:
        r.add(sec, "orchestrator_address", FAIL, str(exc), "Use HOST:PORT, e.g. 192.168.1.10:9090.", tested="config value")
    return cfg


def _http_json(url: str, timeout: float):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def check_orchestrator(r: Report, url: str, expect: Sequence[str], timeout: float) -> Optional[Dict[str, dict]]:
    sec = f"orchestrator {url}"
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http",) or not parsed.hostname or not parsed.port:
        r.add(sec, "url", FAIL, f"{url!r} must look like http://HOST:PORT", tested="parse")
        return None
    probe = tcp_probe(parsed.hostname, parsed.port, timeout)
    r.add(sec, "REST reachable", PASS if probe.ok else FAIL, probe.detail, "" if probe.ok else hint_for(probe),
          tested=f"DNS + TCP connect {parsed.hostname}:{parsed.port}")
    if not probe.ok:
        r.add(sec, "registrations", SKIP, "not tested: the REST port is unreachable")
        return None
    try:
        health = _http_json(f"{url.rstrip('/')}/health", timeout)
        ok = health.get("status") == "ok"
        r.add(sec, "health", PASS if ok else FAIL, json.dumps(health)[:200],
              "" if ok else "Something answered that is not a healthy orchestrator.", tested="GET /health")
        workers = _http_json(f"{url.rstrip('/')}/api/workers", timeout)
    except (urllib.error.URLError, ValueError, OSError) as exc:
        r.add(sec, "health", FAIL, f"HTTP request failed: {exc}",
              "The port is open but did not answer like the orchestrator's REST API (wrong port?).", tested="GET /health")
        return None
    by_id = {w.get("id"): w for w in workers}
    r.add(sec, "registered workers", INFO, ", ".join(
        f"{w['id']} ({'online' if w.get('online') else 'OFFLINE'}, {w.get('profile_source')}, host {w.get('hostname')}, "
        f"reported addr {w.get('address')}, connected from {w.get('observed_peer_ip') or 'unknown'})" for w in workers)
          or "none", tested="GET /api/workers")
    for wid in expect:
        w = by_id.get(wid)
        if w is None:
            r.add(sec, f"worker {wid}", FAIL, "not registered",
                  "Start it (python -m worker.main --config ...) and check its orchestrator_address and the firewall rule "
                  "for the gRPC port (9090) on this orchestrator's machine.", tested="GET /api/workers")
            continue
        problems = []
        if not w.get("online"):
            problems.append(f"offline (last heartbeat {w.get('last_heartbeat_age_seconds')}s ago)")
        if w.get("profile_source") != "REAL":
            problems.append(f"profile_source {w.get('profile_source')}, not REAL")
        if not problems and w.get("backend_probe_pending"):
            r.add(sec, f"worker {wid}", WARN, "online and REAL, but its inference backend probe is still running",
                  "Planning refuses workers with unknown capabilities: wait until the probe finishes (about 10-60 s "
                  "after start; check again).", tested="GET /api/workers")
            continue
        r.add(sec, f"worker {wid}", FAIL if problems else PASS,
              "; ".join(problems) or f"online, REAL, host {w.get('hostname')}, heartbeat {w.get('last_heartbeat_age_seconds'):.1f}s ago",
              "Check the worker's log; a stale worker reconnects automatically when it can reach the orchestrator."
              if problems else "", tested="GET /api/workers")
    return by_id


def get_info(addr: str, timeout: float):
    """GetInfo, or (for a Phase 6 worker) a rejected-version probe that
    executes nothing. Returns (info dict, how)."""
    import grpc

    from partitioner.execution import wire

    pb = wire.pb2()  # also puts worker/ on sys.path
    from worker.generated import stage_execution_pb2_grpc as pb_grpc

    with grpc.insecure_channel(addr) as ch:
        stub = pb_grpc.StageExecutionStub(ch)
        try:
            info = stub.GetInfo(pb.GetInfoRequest(protocol_version=pb.PROTOCOL_VERSION_1), timeout=timeout)
            return {
                "protocol_version": int(info.protocol_version), "supported_versions": [int(v) for v in info.supported_versions],
                "worker_id": info.worker_id, "worker_hostname": info.worker_hostname, "worker_pid": info.worker_pid,
                "executable_model_ids": list(info.executable_model_ids), "supported_encodings": list(info.supported_encodings),
                "max_tensor_bytes": info.max_tensor_bytes, "max_message_bytes": info.max_message_bytes,
                "max_concurrent": info.max_concurrent, "max_batch": info.max_batch, "max_sequence": info.max_sequence,
                "min_deadline_seconds": info.min_deadline_seconds, "authenticated": info.authenticated,
            }, "GetInfo"
        except grpc.RpcError as exc:
            if exc.code() != grpc.StatusCode.UNIMPLEMENTED:
                raise
        # Phase 6 worker: version 0 is rejected before anything else is looked at.
        resp = stub.ExecuteStage(pb.ExecuteStageRequest(protocol_version=pb.PROTOCOL_VERSION_UNSPECIFIED), timeout=timeout)
        return {"protocol_version": int(resp.protocol_version), "supported_versions": [int(resp.protocol_version)],
                "worker_id": resp.worker_id, "worker_hostname": resp.worker_hostname, "worker_pid": resp.worker_pid,
                "probe_status": pb.StageStatus.Name(resp.status), "probe_error": resp.error.code}, "version probe"


def check_endpoint(r: Report, wid: str, addr: str, timeout: float) -> Optional[dict]:
    import grpc

    from partitioner.execution import wire

    sec = f"endpoint {wid}={addr}"
    try:
        host, port = split_host_port(addr)
    except ValueError as exc:
        r.add(sec, "address", FAIL, str(exc), "Use WORKER_ID=IP:PORT, e.g. computer-2=192.168.1.21:50200.")
        return None
    probe = tcp_probe(host, port, timeout)
    r.add(sec, "TCP reachable", PASS if probe.ok else FAIL, probe.detail, "" if probe.ok else hint_for(probe),
          tested=f"DNS + TCP connect {host}:{port} ({probe.seconds * 1e3:.0f} ms)")
    if not probe.ok:
        r.add(sec, "protocol", SKIP, "not tested: TCP connection failed")
        return None
    try:
        info, how = get_info(addr, timeout)
    except grpc.RpcError as exc:
        r.add(sec, "protocol", FAIL, f"TCP is open but the gRPC call failed: {exc.code().name} {exc.details()}",
              "Is this port really the worker's StageExecution port (execution_listen_port), not its Health port "
              "or another program?", tested="gRPC GetInfo")
        return None
    local = wire.PROTOCOL_VERSION
    compatible = local in info["supported_versions"]
    if how == "version probe":
        r.add(sec, "protocol", WARN if compatible else FAIL,
              f"worker predates GetInfo (Phase 6); it reports protocol {info['protocol_version']}, coordinator speaks {local}",
              "Update the worker's checkout to the same commit as the coordinator.",
              tested="ExecuteStage with protocol_version 0 (rejected before execution)")
    else:
        r.add(sec, "protocol", PASS if compatible else FAIL,
              f"worker speaks {info['supported_versions']}, coordinator speaks {local}",
              "" if compatible else "Run the same commit on every machine (git pull, then restart the worker).",
              tested="gRPC GetInfo")
    if info["worker_id"] != wid:
        r.add(sec, "identity", FAIL, f"the endpoint answers as worker {info['worker_id']!r}, not {wid!r}",
              "Wrong IP or port for this worker, or a worker_id typo in its config.", tested="GetInfo worker_id (self-reported)")
    else:
        r.add(sec, "identity", PASS, f"answers as {wid!r} on host {info['worker_hostname']!r} (pid {info['worker_pid']}); "
              "self-reported, unauthenticated", tested="GetInfo worker_id")
    if "executable_model_ids" in info:
        r.add(sec, "worker limits", INFO,
              f"models {info['executable_model_ids']}, max tensor {info['max_tensor_bytes']} B, max message "
              f"{info['max_message_bytes']} B, max batch {info['max_batch']}, max sequence {info['max_sequence'] or 'model max'}, "
              f"max concurrent {info['max_concurrent']}, authenticated {info['authenticated']}", tested="GetInfo")
    return info


def check_plan(r: Report, path: str, endpoints: Dict[str, str], infos: Dict[str, Optional[dict]],
               orch: Optional[Dict[str, dict]], orch_url: Optional[str], require_multi_host: bool) -> None:
    from partitioner.execution.loader import REMOTE_MODE, ExecutionRejected, load_for_execution
    from partitioner.execution.remote import RemoteOptions

    sec = f"plan {path}"
    try:
        loaded = load_for_execution(path, REMOTE_MODE, None)
    except ExecutionRejected as exc:
        for p in exc.problems:
            r.add(sec, "validity", FAIL, p, "Re-plan from live worker records (python -m partitioner plan ...).",
                  tested="Phase 5 loader, remote mode")
        return
    plan, spec = loaded.plan, loaded.spec
    r.add(sec, "validity", PASS, f"plan {plan['inputs_fingerprint'][:16]}... for {spec.model_id}, {len(plan['stages'])} "
          "stage(s); fingerprint, model, precision, ranges and memory recomputation all pass", tested="Phase 5 loader, remote mode")
    if orch_url:
        from partitioner.validate import PlanValidationError, estimator_for_plan, validate_plan
        from partitioner.workers import from_orchestrator

        try:
            records = {w.worker_id: w for w in from_orchestrator(orch_url)}
            validate_plan(plan, spec, records, estimator_for_plan(plan, spec))
            r.add(sec, "capacity vs. live orchestrator", PASS, "stated capacities still fit the workers' current usable RAM",
                  tested="validate_plan with live orchestrator records")
        except PlanValidationError as exc:
            r.add(sec, "capacity vs. live orchestrator", WARN, "; ".join(exc.problems)[:400],
                  "Worker resources changed since planning; re-plan from the live orchestrator.",
                  tested="validate_plan with live orchestrator records")
        except Exception as exc:
            r.add(sec, "capacity vs. live orchestrator", SKIP, f"not tested: {exc}")
    here = socket.gethostname()
    hosts = set()
    for st in plan["stages"]:
        wid, k = st["worker_id"], st["stage_index"]
        name = f"stage {k} -> {wid}"
        hosts.add(st.get("hostname"))
        problems, notes = [], []
        addr = endpoints.get(wid)
        if addr is None:
            r.add(sec, name, FAIL, "no --endpoint given for this worker",
                  f"Add --endpoint {wid}=IP:PORT (its execution_listen_host:execution_listen_port).")
            continue
        try:
            ehost, _ = split_host_port(addr)
            if _classify(ehost) == "loopback" and st.get("hostname") != here:
                problems.append(f"endpoint {addr} is loopback but the stage is planned for host {st.get('hostname')!r}")
        except ValueError:
            pass
        info = infos.get(wid)
        if info is None:
            notes.append("endpoint not verified (unreachable or failed)")
        else:
            if info["worker_id"] != wid:
                problems.append(f"endpoint answers as {info['worker_id']!r}")
            if info["worker_hostname"] != st.get("hostname"):
                problems.append(f"worker reports host {info['worker_hostname']!r}, plan says {st.get('hostname')!r}")
            if "executable_model_ids" in info and spec.model_id not in info["executable_model_ids"]:
                problems.append(f"worker cannot execute {spec.model_id}")
        if orch is not None:
            rec = orch.get(wid)
            if rec is None:
                problems.append("not registered with the orchestrator")
            else:
                if not rec.get("online"):
                    problems.append("offline at the orchestrator")
                if rec.get("hostname") != st.get("hostname"):
                    problems.append(f"orchestrator lists host {rec.get('hostname')!r}, plan says {st.get('hostname')!r}")
        status = FAIL if problems else (SKIP if info is None else PASS)
        rec = (orch or {}).get(wid) or {}
        peer = rec.get("observed_peer_ip")
        try:
            ehost = split_host_port(addr)[0]
        except ValueError:
            ehost = ""
        if peer and ehost and _classify(peer) == "private" and _classify(ehost) == "private" and peer != ehost:
            notes.append(f"note: registered from {peer} but endpoint is {ehost} (fine on a multi-homed machine; "
                         "otherwise a wrong IP)")
        r.add(sec, name, status, "; ".join(problems + notes) or
              f"units {st['unit_range']}, host {st.get('hostname')!r}, endpoint {addr}"
              + (f", registered from {peer}" if peer else "") + ": assignment consistent",
              "The plan, the endpoints and the running workers must agree: re-plan from the live orchestrator and pass "
              "each worker's own execution address." if problems else "",
              tested="plan vs. GetInfo" + (" vs. orchestrator" if orch is not None else ""))
    r.add(sec, "hosts spanned", FAIL if (require_multi_host and len(hosts) < 2) else INFO,
          f"{len(hosts)} host(s): {sorted(h or '?' for h in hosts)}",
          "A physical multi-machine test needs stages on at least two different hosts." if require_multi_host else "",
          tested="plan stage hostnames")

    # ---- limits -----------------------------------------------------
    opts = RemoteOptions()
    wl = plan["workload"]
    batch, tokens = wl["batch_size"], wl["prompt_length"]
    plan_bytes = len(json.dumps(plan).encode("utf-8"))
    for ss, st in zip(loaded.stage_specs, plan["stages"]):
        in_bytes = batch * tokens * (8 if ss.input_kind == "token_ids" else spec.hidden_size * 4)
        out_bytes = batch * tokens * (spec.vocab_size if ss.output_kind == "logits" else spec.hidden_size) * 4
        request_bytes = plan_bytes + in_bytes + 1024
        problems = []
        if out_bytes > opts.max_tensor_bytes or in_bytes > opts.max_tensor_bytes:
            problems.append(f"tensor over the coordinator's {opts.max_tensor_bytes} B default")
        info = infos.get(st["worker_id"])
        tested = "planned workload vs. coordinator defaults"
        if info and "max_tensor_bytes" in info:
            tested += " and worker GetInfo limits"
            max_seq = info["max_sequence"] or spec.max_positions
            if max(in_bytes, out_bytes) > info["max_tensor_bytes"]:
                problems.append(f"tensor {max(in_bytes, out_bytes)} B > worker max_tensor_bytes {info['max_tensor_bytes']}")
            if request_bytes > info["max_message_bytes"]:
                problems.append(f"request ~{request_bytes} B > worker max_message_bytes {info['max_message_bytes']}")
            if batch > info["max_batch"]:
                problems.append(f"planned batch {batch} > worker max_batch {info['max_batch']}")
            if tokens > max_seq:
                problems.append(f"planned prompt_length {tokens} > worker max sequence {max_seq}")
        r.add(sec, f"limits stage {ss.stage_index}", FAIL if problems else PASS,
              "; ".join(problems) or f"batch {batch} x {tokens} tokens: input {in_bytes} B, output {out_bytes} B, request "
              f"~{request_bytes} B -- within limits",
              "Raise the worker's execution_max_* settings or plan a smaller workload (--batch/--prompt)." if problems else "",
              tested=tested)


def run(worker_configs: Sequence[str] = (), orchestrator: Optional[str] = None, expect: Sequence[str] = (),
        endpoints: Optional[Dict[str, str]] = None, plan: Optional[str] = None, require_multi_host: bool = False,
        timeout: float = 3.0, skip_runtime: bool = False, orchestrator_configs: Sequence[str] = (),
        probe_orchestrator: bool = False, peer_ip: str = "") -> Report:
    r = Report()
    endpoints = dict(endpoints or {})
    if not skip_runtime:
        check_runtime(r)
    check_local_network(r)
    for path in orchestrator_configs:
        check_orchestrator_config(r, path, peer_ip or "<COMPUTER-2-IP>")
    for path in worker_configs:
        check_worker_config(r, path, probe_orchestrator, peer_ip or "<COORDINATOR-IP>")
    orch = None
    if orchestrator:
        expected = list(expect)
        if plan and not expected:
            try:
                expected = [st["worker_id"] for st in json.loads(Path(plan).read_text(encoding="utf-8")).get("stages", [])]
            except Exception:
                pass
        orch = check_orchestrator(r, orchestrator, expected, timeout)
    else:
        r.add("orchestrator", "reachability", SKIP, "not tested: no --orchestrator given")
    infos: Dict[str, Optional[dict]] = {}
    for wid, addr in endpoints.items():
        infos[wid] = check_endpoint(r, wid, addr, timeout)
    if not endpoints:
        r.add("endpoints", "reachability", SKIP, "not tested: no --endpoint given; no worker endpoint was contacted")
    if plan:
        check_plan(r, plan, endpoints, infos, orch, orchestrator, require_multi_host)
    else:
        r.add("plan", "validity", SKIP, "not tested: no --plan given")
    return r
