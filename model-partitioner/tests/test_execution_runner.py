"""execute_plan(): single-worker real execution, labelled local
emulation of multi-stage plans, resource limits checked before any
allocation, input validation, and structured errors."""

from __future__ import annotations

import torch

import pytest

from conftest import make_worker, real_local_worker, tiny_plan
from partitioner.execution import tiny_gpt2 as t
from partitioner.execution.interface import Activation, ActivationMeta, STAGE_INTERFACE_VERSION
from partitioner.execution.limits import ExecutionLimits
from partitioner.execution.runner import LOCAL_EMULATION, REAL_MODE, REFERENCE_ATOL, execute_plan
from partitioner.memory import Workload

IDS = [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]]


@pytest.fixture
def local():
    return real_local_worker("local-1")


@pytest.fixture
def single_plan(tiny_spec, local):
    return tiny_plan(tiny_spec, [local], workload=Workload(batch_size=2))


@pytest.fixture
def three_stage_plan(tiny_spec):
    plan = tiny_plan(tiny_spec, [make_worker(f"sim-{i}") for i in range(3)], workload=Workload(batch_size=2),
                     allow_unused_workers=False)
    assert len(plan["stages"]) == 3
    return plan


def run_real(plan, worker, ids=IDS, **kw):
    return execute_plan(plan, ids, mode=REAL_MODE, workers={worker.worker_id: worker}, local_worker_id=worker.worker_id, **kw)


# ---- success -------------------------------------------------------------


def test_single_worker_real_execution(single_plan, local):
    r = run_real(single_plan, local, include_logits=True)
    assert r["status"] == "succeeded", r["error"]
    assert r["execution_mode"] == "local-single-worker" and r["distributed"] is False
    assert r["model"]["model_id"] == "builtin:tiny-gpt2-v1" and r["model"]["weights_version"] == "formula-sin-v1"
    assert r["plan"]["plan_id"] == single_plan["inputs_fingerprint"]
    assert r["plan"]["warnings"] == single_plan["warnings"]  # plan provenance stays visible
    (a,) = r["worker_assignments"]
    assert a["worker_id"] == "local-1" and a["profile_source"] == "REAL" and a["assignment_honored"] is True
    assert r["output"]["shape"] == [2, 8, 128] and r["output"]["dtype"] == "fp32" and r["output"]["kind"] == "logits"
    assert r["output"]["next_token_ids"] == [60, 96]
    ref = t.reference_forward(torch.tensor(IDS))
    assert (torch.tensor(r["output"]["logits"]) - ref).abs().max().item() <= REFERENCE_ATOL
    assert r["reference_check"]["passed"] and r["reference_check"]["max_abs_diff"] <= REFERENCE_ATOL
    assert r["elapsed"]["total_seconds"] > 0 and "MEASURED" in r["elapsed"]["provenance"]
    assert r["stages"][0]["input"] == {"kind": "token_ids", "shape": [2, 8], "dtype": "int64"}
    assert r["handoffs"] == [] and r["error"] is None
    assert r["test_classification"] == "REAL_SINGLE_MACHINE"


def test_multi_stage_plan_emulated_locally_matches_reference(three_stage_plan):
    r = execute_plan(three_stage_plan, IDS, mode=LOCAL_EMULATION, include_logits=True)
    assert r["status"] == "succeeded", r["error"]
    assert r["execution_mode"] == "local-emulation" and r["distributed"] is False
    assert all(a["assignment_honored"] is False for a in r["worker_assignments"])
    assert r["test_classification"] == "SIMULATED"  # the plan's workers are SIMULATED profiles
    assert any("LOCAL EMULATION" in w for w in r["warnings"])
    assert [s["unit_range"] for s in r["stages"]] == [st["unit_range"] for st in three_stage_plan["stages"]]
    assert [h["to_stage"] for h in r["handoffs"]] == [1, 2]
    for s in r["stages"][:-1]:
        assert s["output"] == {"kind": "hidden_states", "shape": [2, 8, 32], "dtype": "fp32"}
    ref = t.reference_forward(torch.tensor(IDS))
    assert (torch.tensor(r["output"]["logits"]) - ref).abs().max().item() <= REFERENCE_ATOL


def test_gpt2_plan_is_never_executed():
    from conftest import REPO

    r = execute_plan(str(REPO / "model-partitioner" / "examples" / "plan_simulated_three_cpu_link.json"), [[1, 2]],
                     mode=LOCAL_EMULATION)
    assert r["status"] == "rejected" and r["stages"] == []
    assert any("'gpt2' has no executable implementation" in p for p in r["error"]["problems"])


# ---- limits: rejected before any allocation -------------------------------


@pytest.fixture
def no_allocation(monkeypatch):
    """Fails the test if weights or the input tensor are created."""
    calls = []

    def forbid(name):
        def f(*a, **k):
            calls.append(name)
            raise AssertionError(f"{name} called despite a limit violation")
        return f

    monkeypatch.setattr(t, "deterministic_param", forbid("deterministic_param"))
    monkeypatch.setattr(t, "token_tensor", forbid("token_tensor"))
    return calls


@pytest.mark.parametrize("ids, limits, message", [
    ([[1, 2]] * 3, ExecutionLimits(max_batch_size=2), "batch size 3 > limit max_batch_size 2"),
    ([[1] * 9], ExecutionLimits(max_sequence_length=8), "sequence length 9 > limit 8"),
    ([[1] * 33], ExecutionLimits(), "sequence length 33 > limit 32"),
    ([[1] * 8], ExecutionLimits(max_activation_bytes=1024), "max_activation_bytes"),
    ([[1] * 8], ExecutionLimits(max_weight_bytes=1000), "max_weight_bytes"),
])
def test_resource_limits_reject_before_allocation(single_plan, local, no_allocation, ids, limits, message):
    r = run_real(single_plan, local, ids=ids, limits=limits)
    assert r["status"] == "rejected" and r["error"]["phase"] == "resource_limits"
    assert any(message in p for p in r["error"]["problems"]), r["error"]["problems"]
    assert r["stages"] == [] and r["output"] is None and no_allocation == []


def test_input_beyond_the_planned_workload_is_rejected(tiny_spec, local, no_allocation):
    plan = tiny_plan(tiny_spec, [local], workload=Workload(batch_size=1, context_length=16, prompt_length=4))
    r = run_real(plan, local, ids=[[1, 2, 3, 4, 5]])
    assert r["status"] == "rejected" and r["error"]["phase"] == "resource_limits"
    assert any("prompt_length 4" in p for p in r["error"]["problems"])
    r = run_real(plan, local, ids=[[1], [2]])
    assert any("workload batch_size 1" in p for p in r["error"]["problems"])


@pytest.mark.parametrize("ids, message", [
    ([], "non-empty list"),
    ([[1, 2], [3]], "same length"),
    ([[1, 128]], r"integers in [0, 128)"),
    ([[1, -1]], r"integers in [0, 128)"),
    ([[1, 2.5]], r"integers in [0, 128)"),
    ([[]], "non-empty list of token ids"),
])
def test_invalid_input_is_rejected_before_allocation(single_plan, local, no_allocation, ids, message):
    r = run_real(single_plan, local, ids=ids)
    assert r["status"] == "rejected" and r["error"]["phase"] == "input_validation"
    assert any(message in p for p in r["error"]["problems"]), r["error"]["problems"]
    assert no_allocation == []


def test_rejection_keeps_plan_provenance(tiny_spec):
    plan = tiny_plan(tiny_spec, [make_worker("sim")])
    r = execute_plan(plan, [[1]], mode=REAL_MODE, workers={}, local_worker_id="sim")
    assert r["status"] == "rejected" and r["error"]["type"] == "ExecutionRejected"
    assert r["plan"]["plan_id"] == plan["inputs_fingerprint"] and r["plan"]["includes_simulated_workers"] is True
    assert r["error"]["problems"] and r["error"]["phase"] == "plan_validation"


# ---- structured errors ---------------------------------------------------


class _Wrapped(t.TinyGPT2Stage):
    """A real stage with one fault injected after construction."""

    fault = None

    def forward(self, activation):
        out = super().forward(activation)
        return type(self).fault(self, out)


def _factory(fault, at_stage):
    def build(stage_spec, spec):
        if stage_spec.stage_index != at_stage:
            return t.TinyGPT2Stage(stage_spec, spec)
        cls = type("Faulty", (_Wrapped,), {"fault": staticmethod(fault)})
        return cls(stage_spec, spec)
    return build


def _raise_with_cause(stage, out):
    try:
        raise ValueError("low-level cause: worker buffer exhausted")
    except ValueError as exc:
        raise RuntimeError("stage blew up") from exc


def _wrong_dtype(stage, out):
    x = out.tensor.double()
    return Activation(ActivationMeta(STAGE_INTERFACE_VERSION, out.meta.request_id, out.meta.kind, tuple(x.shape),
                                     "float64", out.meta.produced_by_stage), x)


def _wrong_shape(stage, out):
    x = out.tensor[:, :-1]
    return Activation(ActivationMeta(STAGE_INTERFACE_VERSION, out.meta.request_id, out.meta.kind, tuple(x.shape),
                                     out.meta.dtype, out.meta.produced_by_stage), x)


def _nan(stage, out):
    x = out.tensor.clone()
    x[0, 0, 0] = float("nan")
    return Activation(out.meta, x)


def _lying_metadata(stage, out):
    return Activation(ActivationMeta(STAGE_INTERFACE_VERSION, out.meta.request_id, out.meta.kind, (1, 1, 1),
                                     out.meta.dtype, out.meta.produced_by_stage), out.tensor)


@pytest.mark.parametrize("fault, error_type, message", [
    (_raise_with_cause, "RuntimeError", "stage blew up"),
    (_wrong_dtype, "BoundaryError", "dtype float64 != expected fp32"),
    (_wrong_shape, "BoundaryError", "shape [2, 7, 32] != expected [2, 8, 32]"),
    (_nan, "BoundaryError", "NaN or infinite"),
    (_lying_metadata, "BoundaryError", "doesn't describe the tensor"),
])
def test_execution_errors_are_structured_results(three_stage_plan, fault, error_type, message):
    r = execute_plan(three_stage_plan, IDS, mode=LOCAL_EMULATION, stage_factory=_factory(fault, at_stage=1))
    assert r["status"] == "failed"
    err = r["error"]
    assert err["type"] == error_type and message in err["message"], err
    assert err["phase"] == "execution" and err["stage_index"] == 1
    assert err["traceback"] and r["output"] is None
    assert [s["stage_index"] for s in r["stages"]] == [0]  # stage 0 completed; nothing after the fault
    if fault is _raise_with_cause:
        assert err["causes"] == [{"type": "ValueError", "message": "low-level cause: worker buffer exhausted"}]


def test_stage_construction_error_is_reported(three_stage_plan):
    def build(stage_spec, spec):
        if stage_spec.stage_index == 2:
            raise MemoryError("could not allocate stage 2 weights")
        return t.TinyGPT2Stage(stage_spec, spec)

    r = execute_plan(three_stage_plan, IDS, mode=LOCAL_EMULATION, stage_factory=build)
    assert r["status"] == "failed" and r["error"]["phase"] == "stage_construction"
    assert r["error"]["type"] == "MemoryError" and r["error"]["stage_index"] == 2 and r["stages"] == []
