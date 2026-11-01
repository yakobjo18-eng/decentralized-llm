"""The validator rejects every kind of broken plan, with a reason."""

from __future__ import annotations

import copy
import datetime as dt

import pytest

from conftest import make_worker
from partitioner import create_plan
from partitioner.cost import CostModel
from partitioner.memory import MemoryEstimator, MemoryPolicy, Precision, Workload
from partitioner.validate import PlanValidationError, validate_plan


@pytest.fixture(scope="module")
def three_stage(gpt2_spec):
    workers = [make_worker("a"), make_worker("b"), make_worker("c")]
    plan = create_plan(gpt2_spec, workers, cost_model=CostModel(throughput_preference="estimated"),
                       allow_unused_workers=False, now=dt.datetime(2026, 10, 31, tzinfo=dt.timezone.utc))
    assert len(plan["stages"]) == 3
    return plan, {w.worker_id: w for w in workers}


def broken(plan, mutate):
    p = copy.deepcopy(plan)
    mutate(p)
    return p


def expect_invalid(plan, spec, message, workers=None, estimator=None):
    with pytest.raises(PlanValidationError) as exc:
        validate_plan(plan, spec, workers, estimator)
    assert any(message in prob for prob in exc.value.problems), exc.value.problems


def test_valid_plan_passes(three_stage, gpt2_spec):
    plan, workers = three_stage
    est = MemoryEstimator(gpt2_spec, Precision(), Workload(), MemoryPolicy())
    validate_plan(plan, gpt2_spec, workers, est)


def _set_range(k, r):
    def m(p):
        p["stages"][k]["unit_range"] = r
    return m


def test_gap_rejected(three_stage, gpt2_spec):
    plan, _ = three_stage
    s1 = plan["stages"][1]["unit_range"]
    expect_invalid(broken(plan, _set_range(1, [s1[0] + 1, s1[1]])), gpt2_spec, "gap or overlap")


def test_overlap_and_duplicate_layers_rejected(three_stage, gpt2_spec):
    plan, _ = three_stage
    s1 = plan["stages"][1]["unit_range"]

    def overlap(p):
        p["stages"][1]["unit_range"] = [s1[0] - 1, s1[1]]
        p["stages"][1]["units"] = [u.name for u in gpt2_spec.units[s1[0] - 1:s1[1]]]
        p["stages"][1]["components"] = [c for u in gpt2_spec.units[s1[0] - 1:s1[1]] for c in u.components]

    bad = broken(plan, overlap)
    expect_invalid(bad, gpt2_spec, "gap or overlap")
    expect_invalid(bad, gpt2_spec, "is assigned 2 times")


def test_missing_layer_rejected(three_stage, gpt2_spec):
    plan, _ = three_stage
    last = plan["stages"][-1]["unit_range"]
    bad = broken(plan, _set_range(2, [last[0], last[1] - 1]))
    expect_invalid(bad, gpt2_spec, "expected 15")


def test_reordered_stages_rejected(three_stage, gpt2_spec):
    plan, _ = three_stage
    bad = broken(plan, lambda p: p["stages"].reverse())
    expect_invalid(bad, gpt2_spec, "stage_index")


def test_component_order_tampering_rejected(three_stage, gpt2_spec):
    plan, _ = three_stage
    bad = broken(plan, lambda p: p["stages"][0]["components"].reverse())
    expect_invalid(bad, gpt2_spec, "order must be preserved")


def test_component_dropped_rejected(three_stage, gpt2_spec):
    plan, _ = three_stage
    bad = broken(plan, lambda p: p["stages"][0]["components"].remove("transformer.wpe"))
    expect_invalid(bad, gpt2_spec, "transformer.wpe is not assigned")


def test_wrong_layer_range_rejected(three_stage, gpt2_spec):
    plan, _ = three_stage
    bad = broken(plan, lambda p: p["stages"][0]["layer_range"].update(last_block=11))
    expect_invalid(bad, gpt2_spec, "layer_range")


def test_duplicate_worker_rejected(three_stage, gpt2_spec):
    plan, _ = three_stage
    bad = broken(plan, lambda p: p["stages"][2].update(worker_id="a"))
    expect_invalid(bad, gpt2_spec, "holds 2 stages")


def test_unknown_worker_rejected(three_stage, gpt2_spec):
    plan, workers = three_stage
    bad = broken(plan, lambda p: p["stages"][1].update(worker_id="ghost"))
    expect_invalid(bad, gpt2_spec, "does not exist", workers=workers)


def test_memory_overcommit_rejected(three_stage, gpt2_spec):
    plan, _ = three_stage
    bad = broken(plan, lambda p: p["stages"][0]["available_memory"].update(device_capacity_bytes=1000))
    expect_invalid(bad, gpt2_spec, "> capacity")


def test_understated_memory_caught_by_recomputation(three_stage, gpt2_spec):
    plan, workers = three_stage
    est = MemoryEstimator(gpt2_spec, Precision(), Workload(), MemoryPolicy())
    bad = broken(plan, lambda p: p["stages"][0]["memory"].update(device_bytes=10))
    expect_invalid(bad, gpt2_spec, "!= recomputed", estimator=est)


def test_inflated_capacity_caught_against_worker(three_stage, gpt2_spec):
    plan, workers = three_stage
    bad = broken(plan, lambda p: p["stages"][0]["available_memory"].update(host_capacity_bytes=10**15))
    expect_invalid(bad, gpt2_spec, "exceeds the worker's usable RAM", workers=workers)


def test_profile_source_lie_rejected(three_stage, gpt2_spec):
    plan, workers = three_stage
    bad = broken(plan, lambda p: p["stages"][0].update(profile_source="REAL"))
    expect_invalid(bad, gpt2_spec, "profile_source", workers=workers)
    expect_invalid(broken(plan, lambda p: p.update(includes_simulated_workers=False)), gpt2_spec,
                   "includes_simulated_workers")


def test_worker_both_used_and_unused_rejected(three_stage, gpt2_spec):
    plan, _ = three_stage
    bad = broken(plan, lambda p: p["unused_workers"].append({"worker_id": "a", "profile_source": "SIMULATED", "reason": "x"}))
    expect_invalid(bad, gpt2_spec, "both assigned a stage and listed as unused")


def test_different_model_config_rejected(three_stage, small_spec):
    plan, _ = three_stage
    expect_invalid(plan, small_spec, "config_fingerprint mismatch")


def test_unsupported_schema_version_rejected(three_stage, gpt2_spec):
    plan, _ = three_stage
    expect_invalid(broken(plan, lambda p: p.update(partition_plan_schema_version=99)), gpt2_spec, "unsupported")


def test_infeasible_plan_with_stages_rejected(three_stage, gpt2_spec):
    plan, _ = three_stage
    bad = broken(plan, lambda p: p.update(feasible=False, infeasibility={"reasons": ["x"]}))
    expect_invalid(bad, gpt2_spec, "must not contain stages")


def test_all_problems_reported(three_stage, gpt2_spec):
    plan, _ = three_stage

    def many(p):
        p["stages"][0]["components"].remove("transformer.wpe")
        p["stages"][2]["worker_id"] = "a"

    with pytest.raises(PlanValidationError) as exc:
        validate_plan(broken(plan, many), gpt2_spec)
    assert len(exc.value.problems) >= 2
