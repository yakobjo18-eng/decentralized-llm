"""Loading Phase 4 plans for execution: valid plans, every rejection
path, simulated-worker refusal in real mode, and the planner's
`executable` flag."""

from __future__ import annotations

import copy
import json

import pytest

from conftest import make_worker, real_local_worker, refingerprint, tiny_plan
from partitioner.execution.loader import LOCAL_EMULATION, REAL_MODE, ExecutionRejected, load_for_execution
from partitioner.memory import Precision


@pytest.fixture
def local():
    return real_local_worker("local-1")


@pytest.fixture
def single_plan(tiny_spec, local):
    return tiny_plan(tiny_spec, [local])


def rejected(source, **kw) -> list:
    with pytest.raises(ExecutionRejected) as exc:
        load_for_execution(source, **kw)
    return exc.value.problems


def real_kw(worker):
    return dict(mode=REAL_MODE, workers={worker.worker_id: worker}, local_worker_id=worker.worker_id)


# ---- planner flag --------------------------------------------------------


def test_planner_marks_only_single_real_tiny_plans_executable(tiny_spec, small_spec, local):
    p = tiny_plan(tiny_spec, [local])
    assert p["executable"] is True and len(p["stages"]) == 1 and p["stages"][0]["unit_range"] == [0, 7]
    sim = tiny_plan(tiny_spec, [make_worker("sim")])
    assert sim["executable"] is False and "SIMULATED" in sim["executable_reason"]
    two = tiny_plan(tiny_spec, [real_local_worker("a"), real_local_worker("b")], allow_unused_workers=False)
    assert len(two["stages"]) == 2 and two["executable"] is False and "activation transport" in two["executable_reason"]
    q = tiny_plan(tiny_spec, [local], precision=Precision("fp32", quantization="int8"))
    assert q["executable"] is False and "quantized execution is not implemented" in q["executable_reason"]
    other = tiny_plan(small_spec, [local])
    assert other["executable"] is False and "no executable implementation" in other["executable_reason"]


# ---- valid ---------------------------------------------------------------


def test_valid_single_stage_plan_loads_in_real_mode(single_plan, local):
    before = copy.deepcopy(single_plan)
    loaded = load_for_execution(single_plan, **real_kw(local))
    assert single_plan == before  # nothing rewritten
    (ss,) = loaded.stage_specs
    assert ss.unit_range == (0, 7) and ss.input_kind == "token_ids" and ss.output_kind == "logits"
    assert loaded.plan_info["plan_id"] == single_plan["inputs_fingerprint"]
    assert loaded.plan_info["warnings"] == single_plan["warnings"] and loaded.warnings == []


def test_valid_plan_loads_from_file(tmp_path, single_plan, local):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(single_plan))
    assert load_for_execution(str(path), **real_kw(local)).plan == single_plan


# ---- malformed / unknown -------------------------------------------------


def test_malformed_json_missing_file_and_non_object(tmp_path, local):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert "not valid JSON" in rejected(str(bad), **real_kw(local))[0]
    assert "could not read plan file" in rejected(str(tmp_path / "missing.json"), **real_kw(local))[0]
    arr = tmp_path / "arr.json"
    arr.write_text("[1, 2]")
    assert "must be a JSON object" in rejected(str(arr), **real_kw(local))[0]


def test_unsupported_schema_version(single_plan, local):
    single_plan["partition_plan_schema_version"] = 99
    assert "unsupported partition_plan_schema_version 99" in rejected(single_plan, **real_kw(local))[0]


def test_edited_plan_fails_the_fingerprint_check(single_plan, local):
    single_plan["workload"]["prompt_length"] = 8
    problems = rejected(single_plan, **real_kw(local))
    assert any("modified after planning" in p for p in problems)


def test_unknown_model_is_refused(small_spec, local):
    plan = tiny_plan(small_spec, [local])  # a valid plan, for a model with no implementation
    problems = rejected(plan, **real_kw(local))
    assert any("has no executable implementation" in p for p in problems)


def test_config_version_mismatch_is_refused(single_plan, local):
    single_plan["model"]["config_fingerprint"] = "0" * 64
    problems = rejected(refingerprint(single_plan), **real_kw(local))
    assert any("different model configuration" in p for p in problems)


def test_adapter_version_mismatch_is_refused(single_plan, local):
    single_plan["model"]["adapter"] = "gpt2 v0"
    assert any("adapter" in p for p in rejected(refingerprint(single_plan), **real_kw(local)))


def test_infeasible_plan_is_refused_with_its_reasons(tiny_spec):
    starved = real_local_worker("starved", ram_usable_gib=0.1)
    plan = tiny_plan(tiny_spec, [starved])
    assert not plan["feasible"]
    problems = rejected(plan, **real_kw(starved))
    assert problems and all(p.startswith("plan is infeasible") for p in problems)


def test_quantized_plan_is_refused(tiny_spec, local):
    plan = tiny_plan(tiny_spec, [local], precision=Precision("fp32", quantization="int4"))
    assert any("quantized execution is not implemented" in p for p in rejected(plan, **real_kw(local)))


# ---- ranges and workers --------------------------------------------------


def test_invalid_ranges_are_refused_not_repaired(tiny_spec):
    workers = [real_local_worker("a"), real_local_worker("b")]
    plan = tiny_plan(tiny_spec, workers, allow_unused_workers=False)
    plan["stages"][0]["unit_range"][1] -= 1  # leaves a gap
    problems = rejected(refingerprint(plan), mode=LOCAL_EMULATION)
    assert any("gap or overlap" in p for p in problems)
    assert any("don't match unit_range" in p for p in problems)


def test_missing_worker_record_is_refused(single_plan, local):
    other = real_local_worker("someone-else")
    problems = rejected(single_plan, mode=REAL_MODE, workers={other.worker_id: other}, local_worker_id=local.worker_id)
    assert any("does not exist" in p for p in problems)
    assert any("not among the supplied worker records" in p for p in problems)


def test_real_mode_needs_worker_records_and_identity(single_plan):
    problems = rejected(single_plan, mode=REAL_MODE)
    assert any("needs the worker records" in p for p in problems)
    assert any("needs this process's worker id" in p for p in problems)


def test_stage_on_another_host_is_refused(tiny_spec):
    remote = real_local_worker("remote", hostname="computer-2")
    plan = tiny_plan(tiny_spec, [remote])
    problems = rejected(plan, **real_kw(remote))
    assert any("planned for host 'computer-2'" in p for p in problems)


def test_multi_stage_plan_is_refused_in_real_mode(tiny_spec):
    a, b = real_local_worker("a"), real_local_worker("b")
    plan = tiny_plan(tiny_spec, [a, b], allow_unused_workers=False)
    problems = rejected(plan, mode=REAL_MODE, workers={"a": a, "b": b}, local_worker_id="a")
    assert any("activation transport is not implemented" in p for p in problems)


# ---- simulated workers ---------------------------------------------------


def test_simulated_plan_is_refused_in_real_mode(tiny_spec):
    sims = [make_worker("sim-a"), make_worker("sim-b")]
    plan = tiny_plan(tiny_spec, sims, allow_unused_workers=False)
    assert plan["includes_simulated_workers"]
    problems = rejected(plan, mode=REAL_MODE, workers={w.worker_id: w for w in sims}, local_worker_id="sim-a")
    assert any("SIMULATED workers: refusing real execution" in p for p in problems)
    assert any("is SIMULATED, not REAL" in p for p in problems)


def test_simulated_plan_is_emulated_only_with_explicit_labels(tiny_spec):
    sims = [make_worker("sim-a"), make_worker("sim-b")]
    plan = tiny_plan(tiny_spec, sims, allow_unused_workers=False)
    loaded = load_for_execution(plan, mode=LOCAL_EMULATION, workers={w.worker_id: w for w in sims})
    text = " ".join(loaded.warnings)
    assert "LOCAL EMULATION" in text and "NOT honored" in text and "not distributed" in text
    assert "SIMULATED - NOT PHYSICALLY VERIFIED" in text
    assert loaded.plan_info["includes_simulated_workers"] is True
