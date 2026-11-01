"""Phase 3 worker config keys and LAN address advertisement."""

from __future__ import annotations

from worker.config.loader import load
from worker.main import _own_advertised_address


def _clear_env(monkeypatch):
    for name in (
        "WORKER_GPU_METRICS_INTERVAL_SECONDS",
        "WORKER_BACKEND_COMPUTE_TEST",
        "WORKER_SIMULATED_PROFILE",
        "WORKER_ADVERTISE_HOST",
        "WORKER_LISTEN_HOST",
    ):
        monkeypatch.delenv(name, raising=False)


def test_phase3_defaults(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    cfg = load(str(tmp_path / "missing.yaml"), env_file=str(tmp_path / "no.env"))
    assert cfg.gpu_metrics_interval_seconds == 15.0
    assert cfg.backend_compute_test is True
    assert cfg.simulated_profile == ""  # real hardware unless explicitly configured
    assert cfg.advertise_host == ""


def test_phase3_keys_from_file_and_env(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    f = tmp_path / "worker.yaml"
    f.write_text(
        "gpu_metrics_interval_seconds: 30\nbackend_compute_test: false\n"
        "simulated_profile: profiles/simulated/profile_a_cpu_worker.yaml\nadvertise_host: 192.168.1.20\n"
    )
    cfg = load(str(f), env_file=str(tmp_path / "no.env"))
    assert cfg.gpu_metrics_interval_seconds == 30
    assert cfg.backend_compute_test is False
    assert cfg.simulated_profile.endswith("profile_a_cpu_worker.yaml")
    assert cfg.advertise_host == "192.168.1.20"

    monkeypatch.setenv("WORKER_BACKEND_COMPUTE_TEST", "yes")
    monkeypatch.setenv("WORKER_ADVERTISE_HOST", "10.0.0.7")
    cfg = load(str(f), env_file=str(tmp_path / "no.env"))
    assert cfg.backend_compute_test is True
    assert cfg.advertise_host == "10.0.0.7"


def test_advertised_address_never_assumes_localhost(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    f = tmp_path / "worker.yaml"
    f.write_text('listen_host: "0.0.0.0"\nlisten_port: 50100\n')
    cfg = load(str(f), env_file=str(tmp_path / "no.env"))
    host, port = _own_advertised_address(cfg).rsplit(":", 1)
    assert port == "50100"
    assert host not in ("0.0.0.0", "127.0.0.1", "localhost")

    cfg.advertise_host = "192.168.1.20"
    assert _own_advertised_address(cfg) == "192.168.1.20:50100"

    cfg.advertise_host = ""
    cfg.listen_host = "192.168.1.21"
    assert _own_advertised_address(cfg) == "192.168.1.21:50100"
