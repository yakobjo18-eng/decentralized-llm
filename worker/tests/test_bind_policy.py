"""Phase 7: every worker listener is loopback by default, and the Health
server refuses public / hostname binds at startup (0.0.0.0 only with a
warning)."""

from __future__ import annotations

import logging

import pytest

from worker.config.bind_policy import BindAddressError, check_bind_host, classify
from worker.config.loader import load


def test_health_listen_host_defaults_to_loopback(tmp_path, monkeypatch):
    monkeypatch.delenv("WORKER_LISTEN_HOST", raising=False)
    cfg = load(str(tmp_path / "missing.yaml"), env_file=None)
    assert cfg.listen_host == "127.0.0.1" and cfg.execution_listen_host == "127.0.0.1"


@pytest.mark.parametrize("host, kind", [
    ("127.0.0.1", "loopback"), ("localhost", "loopback"), ("::1", "loopback"), ("192.168.0.199", "private"),
    ("169.254.1.1", "private"), ("0.0.0.0", "all-interfaces"), ("::", "all-interfaces"), ("8.8.8.8", "public"),
    ("my-pc", "hostname"),
])
def test_classify(host, kind):
    assert classify(host) == kind


def test_health_policy_warns_for_all_interfaces(caplog):
    with caplog.at_level(logging.WARNING, logger="worker"):
        check_bind_host("0.0.0.0", allow_all_interfaces=True, setting="listen_host", allow_flag="")
    assert any("listen_host 0.0.0.0 listens on ALL interfaces" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("host, fragment", [("8.8.8.8", "public address"), ("my-pc.example.com", "must be an IP address")])
def test_worker_main_refuses_unsafe_health_bind_before_listening(tmp_path, monkeypatch, host, fragment):
    import worker.main as main

    created = []
    monkeypatch.setattr(main, "create_server", lambda *a, **k: created.append(a))
    monkeypatch.setattr(main.signal, "signal", lambda *a, **k: None)  # keep pytest's own Ctrl+C handling
    cfg = tmp_path / "w.yaml"
    cfg.write_text(f'worker_id: "b"\nlisten_host: "{host}"\n')
    monkeypatch.delenv("WORKER_LISTEN_HOST", raising=False)
    with pytest.raises(BindAddressError, match=fragment):
        main.run(str(cfg))
    assert created == []  # refused before any server was created
