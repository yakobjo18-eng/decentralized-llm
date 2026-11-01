"""CLI: plan the built-in model, then `execute` it -- success and every
failure exit path."""

from __future__ import annotations

import json

import pytest

from conftest import PROFILES, REPO, real_local_worker
from partitioner import cli
from partitioner.cli import main

TINY = "builtin:tiny-gpt2-v1"
SIM = [str(PROFILES / f) for f in ("profile_a_cpu_worker.yaml", "profile_c_high_memory_worker.yaml")]


@pytest.fixture
def fake_local(monkeypatch):
    """Replaces the ~10 s Phase 3 profiling of this machine with a
    synthetic REAL record for this host (test input, not a measurement)."""
    seen = {}

    def profile(worker_id, margins=None, run_compute_test=True):
        seen["margins"] = margins
        return real_local_worker(worker_id)

    monkeypatch.setattr(cli, "from_local_machine", profile)
    return seen


def run(capsys, argv):
    rc = main(argv)
    cap = capsys.readouterr()
    return rc, cap.out, cap.err


def plan_file(capsys, tmp_path, *workers):
    out = tmp_path / "plan.json"
    rc, text, _ = run(capsys, ["plan", "--model", TINY, *workers, "--throughput", "estimated", "--batch", "2",
                               "--out", str(out)])
    assert rc == 0, text
    return out


def test_real_single_worker_execute(capsys, tmp_path, fake_local):
    plan = plan_file(capsys, tmp_path, "--local", "computer-1")
    assert json.loads(plan.read_text())["executable"] is True
    rc, out, err = run(capsys, ["execute", str(plan), "--tokens", "1,2,3,4,5,6,7,8;9,10,11,12,13,14,15,16",
                                "--local", "computer-1", "--out", str(tmp_path / "result.json")])
    assert rc == 0, out
    result = json.loads(out)
    assert result["status"] == "succeeded" and result["execution_mode"] == "local-single-worker"
    assert result["output"]["shape"] == [2, 8, 128] and result["output"]["next_token_ids"] == [60, 96]
    assert result["reference_check"]["passed"] and result["distributed"] is False
    assert json.loads((tmp_path / "result.json").read_text())["request_id"] == result["request_id"]
    assert "SUCCEEDED (local-single-worker, distributed: false)" in err


def test_emulated_execute_of_simulated_plan(capsys, tmp_path):
    plan = plan_file(capsys, tmp_path, "--simulated", *SIM, "--require-all-workers")
    rc, out, _ = run(capsys, ["execute", str(plan), "--tokens", "5,6,7", "--emulate-stages-locally"])
    assert rc == 0, out
    result = json.loads(out)
    assert result["execution_mode"] == "local-emulation" and len(result["stages"]) == 2
    assert any("LOCAL EMULATION" in w for w in result["warnings"])


def test_simulated_plan_rejected_in_real_mode(capsys, tmp_path, fake_local):
    plan = plan_file(capsys, tmp_path, "--simulated", *SIM)
    rc, out, err = run(capsys, ["execute", str(plan), "--tokens", "1,2", "--local", "computer-1"])
    assert rc == 1
    result = json.loads(out)
    assert result["status"] == "rejected"
    assert any("SIMULATED workers: refusing real execution" in p for p in result["error"]["problems"])
    assert "REJECTED (local-single-worker, distributed: false): plan_validation: plan includes SIMULATED" in err


def test_gpt2_plan_rejected(capsys):
    rc, out, _ = run(capsys, ["execute", str(REPO / "model-partitioner" / "examples" / "plan_computer1_plus_simulated.json"),
                              "--tokens", "1,2", "--emulate-stages-locally"])
    assert rc == 1 and json.loads(out)["status"] == "rejected"


def test_limit_violation_exit_code(capsys, tmp_path, fake_local):
    plan = plan_file(capsys, tmp_path, "--local", "computer-1")
    rc, out, _ = run(capsys, ["execute", str(plan), "--tokens", "1,2,3,4,5", "--local", "computer-1", "--max-seq", "4"])
    assert rc == 1
    result = json.loads(out)
    assert result["error"]["phase"] == "resource_limits" and result["stages"] == []


def test_margin_flags_reach_the_local_profiler(capsys, tmp_path, fake_local):
    plan_file(capsys, tmp_path, "--local", "computer-1", "--ram-margin-percent", "5", "--ram-margin-min-gib", "0.25")
    assert (fake_local["margins"].ram_percent, fake_local["margins"].ram_min_gib) == (5.0, 0.25)


@pytest.mark.parametrize("extra", [[], ["--local", "x", "--emulate-stages-locally"]])
def test_execute_needs_exactly_one_mode(capsys, tmp_path, extra):
    rc, _, err = run(capsys, ["execute", str(tmp_path / "p.json"), "--tokens", "1", *extra])
    assert rc == 2 and "choose exactly one" in err


def test_unparseable_tokens_are_a_usage_error(capsys, tmp_path):
    rc, _, err = run(capsys, ["execute", str(tmp_path / "p.json"), "--tokens", "1,two", "--emulate-stages-locally"])
    assert rc == 2 and "--tokens must be" in err


def test_missing_plan_file_is_a_structured_rejection(capsys, tmp_path):
    rc, out, _ = run(capsys, ["execute", str(tmp_path / "nope.json"), "--tokens", "1", "--emulate-stages-locally"])
    assert rc == 1
    result = json.loads(out)
    assert result["status"] == "rejected" and result["error"]["phase"] == "plan_loading"


def test_validate_accepts_the_builtin_model(capsys, tmp_path):
    plan = plan_file(capsys, tmp_path, "--simulated", *SIM)
    rc, out, _ = run(capsys, ["validate", str(plan), "--model", TINY, "--simulated", *SIM])
    assert rc == 0 and "VALID" in out
