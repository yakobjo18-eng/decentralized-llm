"""Phase 6 worker settings: StageExecution is off by default, loopback
by default, configurable from YAML and env, and refuses unsafe binds.
(No torch needed: the service itself is tested in model-partitioner/tests.)"""

from __future__ import annotations

import pytest

from worker.config.loader import load
from worker.execution.server import BindAddressError, check_bind_host


def test_defaults_are_off_and_loopback(tmp_path, monkeypatch):
    for k in ("WORKER_EXECUTION_ENABLED", "WORKER_EXECUTION_LISTEN_HOST", "WORKER_EXECUTION_LISTEN_PORT"):
        monkeypatch.delenv(k, raising=False)
    cfg = load(str(tmp_path / "missing.yaml"), env_file=None)
    assert cfg.execution_enabled is False
    assert (cfg.execution_listen_host, cfg.execution_listen_port) == ("127.0.0.1", 50200)
    assert cfg.execution_allow_all_interfaces is False
    assert cfg.execution_max_concurrent == 2 and cfg.execution_max_tensor_mib == 16.0


def test_yaml_and_env(tmp_path, monkeypatch):
    path = tmp_path / "w.yaml"
    path.write_text('execution_enabled: true\nexecution_listen_host: "192.168.1.20"\nexecution_listen_port: 50300\n'
                    "execution_max_concurrent: 4\nexecution_max_batch: 2\nexecution_min_deadline_seconds: 0.5\n")
    cfg = load(str(path), env_file=None)
    assert cfg.execution_enabled and cfg.execution_listen_host == "192.168.1.20" and cfg.execution_listen_port == 50300
    assert cfg.execution_max_concurrent == 4 and cfg.execution_max_batch == 2 and cfg.execution_min_deadline_seconds == 0.5
    monkeypatch.setenv("WORKER_EXECUTION_LISTEN_HOST", "10.0.0.5")
    monkeypatch.setenv("WORKER_EXECUTION_LISTEN_PORT", "50301")
    monkeypatch.setenv("WORKER_EXECUTION_ENABLED", "false")
    cfg = load(str(path), env_file=None)
    assert (cfg.execution_listen_host, cfg.execution_listen_port, cfg.execution_enabled) == ("10.0.0.5", 50301, False)


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "192.168.1.20", "10.1.2.3", "172.16.0.9", "169.254.3.4",
                                  "fd00::1"])
def test_loopback_and_private_binds_are_allowed(host):
    check_bind_host(host)


@pytest.mark.parametrize("host", ["8.8.8.8", "1.1.1.1", "2001:4860::1"])  # globally routable
def test_public_binds_are_refused(host):
    with pytest.raises(BindAddressError, match="public address"):
        check_bind_host(host, allow_all_interfaces=True)


@pytest.mark.parametrize("host", ["0.0.0.0", "::"])
def test_all_interfaces_needs_explicit_opt_in(host):
    with pytest.raises(BindAddressError, match="EVERY interface"):
        check_bind_host(host)
    check_bind_host(host, allow_all_interfaces=True)


def test_hostnames_are_refused():
    with pytest.raises(BindAddressError, match="must be an IP address"):
        check_bind_host("my-pc.example.com")


def test_simulated_worker_cannot_start_execution(tmp_path):
    from worker.execution.server import start_from_config

    path = tmp_path / "w.yaml"
    path.write_text('worker_id: "s"\nsimulated_profile: "profiles/simulated/profile_a_cpu_worker.yaml"\n')
    with pytest.raises(RuntimeError, match="SIMULATED"):
        start_from_config(load(str(path), env_file=None))
