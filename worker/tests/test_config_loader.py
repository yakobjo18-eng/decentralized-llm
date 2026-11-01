from worker.config.loader import load


def test_default_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("WORKER_ID", raising=False)
    monkeypatch.delenv("WORKER_ORCHESTRATOR_ADDRESS", raising=False)

    cfg = load(str(tmp_path / "does-not-exist.yaml"), env_file=str(tmp_path / "no.env"))

    assert cfg.orchestrator_address == "127.0.0.1:9090"
    assert cfg.listen_port == 50100
    assert cfg.worker_id  # auto-generated, must be non-empty


def test_file_overrides_defaults(tmp_path):
    config_file = tmp_path / "worker.yaml"
    config_file.write_text(
        "worker_id: test-worker\norchestrator_address: 10.0.0.5:9090\nlisten_port: 60000\n"
    )

    cfg = load(str(config_file), env_file=str(tmp_path / "no.env"))

    assert cfg.worker_id == "test-worker"
    assert cfg.orchestrator_address == "10.0.0.5:9090"
    assert cfg.listen_port == 60000
    # untouched by the file -> still the default
    assert cfg.heartbeat_interval_seconds == 5.0


def test_env_overrides_file(tmp_path, monkeypatch):
    config_file = tmp_path / "worker.yaml"
    config_file.write_text("worker_id: from-file\n")
    monkeypatch.setenv("WORKER_ID", "from-env")

    cfg = load(str(config_file), env_file=str(tmp_path / "no.env"))

    assert cfg.worker_id == "from-env"


def test_listen_address_combines_host_and_port(tmp_path):
    config_file = tmp_path / "worker.yaml"
    config_file.write_text("listen_host: 192.168.1.50\nlisten_port: 50200\n")

    cfg = load(str(config_file), env_file=str(tmp_path / "no.env"))

    assert cfg.listen_address == "192.168.1.50:50200"
