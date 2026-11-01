"""CLI: plan from simulated profiles, save, validate the saved file."""

from __future__ import annotations

import json

from conftest import PROFILES
from partitioner.cli import main


def test_plan_then_validate(tmp_path, capsys):
    out = tmp_path / "plan.json"
    profiles = [str(PROFILES / f) for f in ("profile_a_cpu_worker.yaml", "profile_c_high_memory_worker.yaml")]
    rc = main(["plan", "--model", "gpt2", "--simulated", *profiles, "--dtype", "fp16", "--context", "512",
               "--throughput", "estimated", "--out", str(out)])
    text = capsys.readouterr().out
    assert rc == 0, text
    assert "SIMULATED - NOT PHYSICALLY VERIFIED" in text
    plan = json.loads(out.read_text())
    assert plan["feasible"] and plan["precision"]["weight_dtype"] == "fp16" and plan["workload"]["context_length"] == 512

    rc = main(["validate", str(out), "--model", "gpt2", "--simulated", *profiles])
    assert rc == 0
    assert "VALID" in capsys.readouterr().out

    plan["stages"][0]["unit_range"][1] += 1
    out.write_text(json.dumps(plan))
    rc = main(["validate", str(out), "--model", "gpt2"])
    assert rc == 1
    assert "invalid partition plan" in capsys.readouterr().out


def test_memory_command(capsys):
    assert main(["memory", "--model", "gpt2", "--no-checkpoint-check", "--dtype", "bf16"]) == 0
    text = capsys.readouterr().out
    assert "whole model on ONE worker" in text and "MEASURED" in text and "CALCULATED" in text


def test_no_workers_is_a_usage_error(capsys):
    assert main(["plan", "--model", "gpt2", "--no-checkpoint-check"]) == 2


def test_infeasible_plan_exit_code(capsys, tmp_path):
    # Profile A with an unrealistically large safety margin leaves too little memory.
    rc = main(["plan", "--model", "gpt2", "--no-checkpoint-check", "--simulated", str(PROFILES / "profile_a_cpu_worker.yaml"),
               "--extra-safety", "0.95", "--out", str(tmp_path / "p.json")])
    assert rc == 1
    assert "INFEASIBLE" in capsys.readouterr().out
