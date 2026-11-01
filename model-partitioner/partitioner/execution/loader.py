"""Load a Phase 4 plan for execution, or refuse it with every reason.

Nothing is repaired or rewritten: the plan is checked as written.

Checks, in addition to partitioner.validate.validate_plan (coverage,
contiguity, order, workers, source labels, memory recomputation, config
fingerprint):
  - the file parses, is an object, and has a supported schema version;
  - its content still matches its inputs_fingerprint (not edited after
    planning);
  - it is feasible;
  - its model has an executable implementation (models.py) and the
    adapter version matches;
  - its precision is executable (fp32, no quantization).

Modes:
  REAL             -- every stage must be on a REAL worker whose record
                      is supplied, the plan must say it is executable,
                      and every stage must be assigned to THIS process's
                      worker on THIS host. With no activation transport,
                      that means a single-stage plan.
  LOCAL_EMULATION  -- explicit opt-in: every stage runs sequentially in
                      this one process, whatever worker it is assigned
                      to (SIMULATED included). Results say so; this is
                      never distributed execution.
"""

from __future__ import annotations

import dataclasses
import json
import socket
from pathlib import Path
from typing import List, Mapping, Optional, Tuple, Union

from partitioner import plan as plan_mod
from partitioner.execution.dtypes import EXECUTABLE_DTYPES
from partitioner.execution.interface import HIDDEN_STATES, LOGITS, STAGE_INTERFACE_VERSION, TOKEN_IDS, StageSpec
from partitioner.execution.models import EXECUTABLE_MODEL_IDS, ExecutableModel, get_executable_model
from partitioner.validate import PlanValidationError, estimator_for_plan, validate_plan
from partitioner.workers import REAL, SIMULATED

REAL_MODE = "real"
LOCAL_EMULATION = "local-emulation"
MODES = (REAL_MODE, LOCAL_EMULATION)


class ExecutionRejected(Exception):
    """The plan (or request) must not be executed. `problems` lists every
    reason found; `plan_info` carries the plan's provenance if it could
    be read."""

    def __init__(self, problems: List[str], plan_info: Optional[dict] = None, phase: str = "plan_validation"):
        self.problems = problems
        self.plan_info = plan_info
        self.phase = phase
        super().__init__("execution rejected:\n  - " + "\n  - ".join(problems))


@dataclasses.dataclass
class LoadedPlan:
    plan: dict
    plan_info: dict
    model: ExecutableModel
    spec: object  # ModelSpec
    mode: str
    stage_specs: Tuple[StageSpec, ...]
    warnings: List[str]


def plan_info(plan: dict) -> dict:
    """Provenance carried into every execution result."""
    return {
        "plan_id": plan.get("inputs_fingerprint"),
        "plan_id_basis": "inputs_fingerprint (sha256 of the plan content)",
        "partition_plan_schema_version": plan.get("partition_plan_schema_version"),
        "created_at": plan.get("created_at"),
        "planner": plan.get("planner"),
        "feasible": plan.get("feasible"),
        "includes_simulated_workers": plan.get("includes_simulated_workers"),
        "executable": plan.get("executable"),
        "executable_reason": plan.get("executable_reason"),
        "warnings": list(plan.get("warnings") or []),
        "assumptions": list(plan.get("assumptions") or []),
    }


def read_plan(source: Union[str, Path, dict]) -> dict:
    if isinstance(source, dict):
        return source
    try:
        text = Path(source).read_text(encoding="utf-8")
    except OSError as exc:
        raise ExecutionRejected([f"could not read plan file {str(source)!r}: {exc}"], phase="plan_loading") from exc
    try:
        plan = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ExecutionRejected([f"plan file {str(source)!r} is not valid JSON: {exc}"], phase="plan_loading") from exc
    if not isinstance(plan, dict):
        raise ExecutionRejected([f"plan must be a JSON object, got {type(plan).__name__}"], phase="plan_loading")
    return plan


def _stage_spec(plan: dict, st: dict, spec, n_stages: int) -> StageSpec:
    """Built from what the PLAN says; the executor independently checks
    it against the model."""
    start, end = st["unit_range"]
    last_kind = spec.units[end - 1].kind if 0 < end <= len(spec.units) else None
    return StageSpec(
        interface_version=STAGE_INTERFACE_VERSION,
        model_id=plan["model"]["model_id"],
        config_fingerprint=plan["model"]["config_fingerprint"],
        stage_index=st["stage_index"],
        n_stages=n_stages,
        unit_range=(start, end),
        unit_names=tuple(st.get("units") or ()),
        components=tuple(st.get("components") or ()),
        input_kind=TOKEN_IDS if start == 0 else HIDDEN_STATES,
        output_kind=LOGITS if last_kind == "lm_head" else HIDDEN_STATES,
        activation_dtype=plan["precision"]["activation_dtype"],
    )


def load_for_execution(
    source: Union[str, Path, dict],
    mode: str = REAL_MODE,
    workers: Optional[Mapping[str, object]] = None,
    local_worker_id: Optional[str] = None,
    local_hostname: Optional[str] = None,
) -> LoadedPlan:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    plan = read_plan(source)
    version = plan.get("partition_plan_schema_version")
    if version not in plan_mod.SUPPORTED_SCHEMA_VERSIONS:
        raise ExecutionRejected([f"unsupported partition_plan_schema_version {version!r} "
                                 f"(supported: {list(plan_mod.SUPPORTED_SCHEMA_VERSIONS)})"])
    info = plan_info(plan)
    problems: List[str] = []
    warnings: List[str] = []

    stated = plan.get("inputs_fingerprint")
    if stated != plan_mod.fingerprint(plan):
        problems.append("plan content does not match its inputs_fingerprint: it was modified after planning "
                        "(or the fingerprint is missing); re-plan instead of editing")
    if not plan.get("feasible"):
        reasons = (plan.get("infeasibility") or {}).get("reasons") or ["no reasons given"]
        raise ExecutionRejected(problems + ["plan is infeasible: " + r for r in reasons], info)

    model_section = plan.get("model")
    model_id = model_section.get("model_id") if isinstance(model_section, dict) else None
    model = get_executable_model(model_id) if isinstance(model_id, str) else None
    if model is None:
        raise ExecutionRejected(problems + [
            f"model {model_id!r} has no executable implementation (executable models: {list(EXECUTABLE_MODEL_IDS)}); "
            "GPT-2 small and other models are plannable only"
        ], info)
    spec = model.spec()
    adapter = f"{spec.adapter} v{spec.adapter_version}"
    if model_section.get("adapter") != adapter:
        problems.append(f"plan adapter {model_section.get('adapter')!r} != executable model's {adapter!r}")

    precision = plan.get("precision") if isinstance(plan.get("precision"), dict) else {}
    if precision.get("quantization") != "none":
        problems.append(f"quantization {precision.get('quantization')!r}: quantized execution is not implemented")
    for field in ("weight_dtype", "activation_dtype"):
        if precision.get(field) not in EXECUTABLE_DTYPES:
            problems.append(f"{field} {precision.get(field)!r} is not executable (supported: {list(EXECUTABLE_DTYPES)})")

    estimator = None
    try:
        estimator = estimator_for_plan(plan, spec)
    except Exception as exc:  # malformed precision/workload/memory_policy sections
        problems.append(f"memory settings in the plan are unusable, so its memory can't be recomputed: "
                        f"{type(exc).__name__}: {exc}")
    try:
        validate_plan(plan, spec, workers, estimator)
    except PlanValidationError as exc:
        problems.extend(exc.problems)
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        problems.append(f"plan is malformed: {type(exc).__name__}: {exc}")

    stages = plan.get("stages") if isinstance(plan.get("stages"), list) else []
    sources = [st.get("profile_source") for st in stages if isinstance(st, dict)]
    host = local_hostname or socket.gethostname()
    n_general = len(problems)
    if mode == REAL_MODE:
        if workers is None:
            problems.append("real execution needs the worker records the plan is checked against (none given)")
        if plan.get("includes_simulated_workers") or SIMULATED in sources:
            problems.append("plan includes SIMULATED workers: refusing real execution (SIMULATED profiles are not "
                            "physical machines; use local-emulation mode to run it in one process, labelled as such)")
        if plan.get("executable") is not True:
            problems.append(f"plan is not marked executable: {plan.get('executable_reason')!r}")
        if not local_worker_id:
            problems.append("real execution needs this process's worker id (local_worker_id)")
        else:
            w = (workers or {}).get(local_worker_id)
            if workers is not None and w is None:
                problems.append(f"this process's worker {local_worker_id!r} is not among the supplied worker records")
            elif w is not None and getattr(w, "profile_source", None) != REAL:
                problems.append(f"this process's worker {local_worker_id!r} is {getattr(w, 'profile_source', None)}, not REAL")
            for st in stages:
                if not isinstance(st, dict):
                    continue
                if st.get("worker_id") != local_worker_id:
                    problems.append(
                        f"stage {st.get('stage_index')} is assigned to worker {st.get('worker_id')!r}, but this process is "
                        f"{local_worker_id!r}: remote stage execution / activation transport is not implemented"
                    )
                elif st.get("hostname") != host:
                    problems.append(f"stage {st.get('stage_index')} was planned for host {st.get('hostname')!r}, "
                                    f"this is {host!r}")
    else:
        warnings.append(
            f"LOCAL EMULATION: all {len(stages)} stage(s) run sequentially in this one process on {host}. The plan's "
            "worker assignments are NOT honored; this is not distributed execution and says nothing about "
            "multi-machine behaviour."
        )
        if plan.get("includes_simulated_workers") or SIMULATED in sources:
            warnings.append("The plan uses SIMULATED - NOT PHYSICALLY VERIFIED workers; only its stage boundaries are used.")
        if plan.get("executable") is not True:
            warnings.append(f"The plan is not marked executable ({plan.get('executable_reason')}); emulated anyway on request.")
        if workers is None:
            warnings.append("Worker records were not supplied, so worker existence and capacity were not checked.")

    if problems:
        # Mode-level reasons (SIMULATED, remote stage, ...) first: they
        # are usually the root cause of the validator's findings.
        raise ExecutionRejected(problems[n_general:] + problems[:n_general], info)
    stage_specs = tuple(_stage_spec(plan, st, spec, len(stages)) for st in stages)
    return LoadedPlan(plan, info, model, spec, mode, stage_specs, warnings)
