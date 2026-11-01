"""Registry of models that can actually be EXECUTED (Phase 5).

Planning works for any model an adapter describes; execution needs a
real implementation and weights. The only registered model is the
built-in test model builtin:tiny-gpt2-v1. GPT-2 small (and anything
else) is plannable but not executable: its weights are never downloaded
or allocated by this project.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Optional, Sequence, Tuple

from partitioner.execution.dtypes import EXECUTABLE_DTYPES

TINY_GPT2_ID = "builtin:tiny-gpt2-v1"
EXECUTABLE_MODEL_IDS = (TINY_GPT2_ID,)

# Defence in depth: refuse to materialize more weight bytes than this
# for any registered model, whatever its config says.
MAX_EXECUTABLE_WEIGHT_BYTES = 64 * 2**20


@dataclasses.dataclass(frozen=True)
class ExecutableModel:
    model_id: str
    description: str
    weights_version: str
    spec: Callable  # () -> ModelSpec
    build_stage: Callable  # (StageSpec, ModelSpec) -> StageExecutor
    reference_forward: Callable  # (token_ids tensor) -> logits tensor
    weight_bytes: Callable  # (ModelSpec, start, end) -> int, before allocating
    token_tensor: Callable  # (list of lists) -> int64 tensor


def get_executable_model(model_id: str) -> Optional[ExecutableModel]:
    if model_id != TINY_GPT2_ID:
        return None
    from partitioner.execution import tiny_gpt2 as t

    return ExecutableModel(
        model_id=t.MODEL_ID,
        description=t.DESCRIPTION,
        weights_version=t.WEIGHTS_VERSION,
        spec=t.spec,
        build_stage=lambda stage_spec, model_spec: t.TinyGPT2Stage(stage_spec, model_spec),
        reference_forward=t.reference_forward,
        weight_bytes=t.weight_bytes,
        token_tensor=t.token_tensor,
    )


def executable_status(model_id: str, precision, stage_sources: Sequence[str]) -> Tuple[bool, str]:
    """The planner's `executable` flag for a FEASIBLE plan. The execution
    loader re-checks all of this itself and never trusts the flag alone."""
    if model_id not in EXECUTABLE_MODEL_IDS:
        return False, (
            f"Phase 4 produces plans only for this model: no executable implementation exists for {model_id!r} "
            f"(Phase 5 executes only {list(EXECUTABLE_MODEL_IDS)})"
        )
    reasons = []
    if precision.quantization != "none":
        reasons.append(f"{precision.quantization} quantized execution is not implemented")
    if precision.weight_dtype not in EXECUTABLE_DTYPES or precision.act not in EXECUTABLE_DTYPES:
        reasons.append(f"only {list(EXECUTABLE_DTYPES)} execution is implemented")
    if "SIMULATED" in stage_sources:
        reasons.append("the plan uses SIMULATED workers (it can only be emulated locally)")
    if len(stage_sources) > 1:
        reasons.append(
            "a multi-stage plan needs activation transport between workers, which is not implemented "
            "(it can only be emulated in one process)"
        )
    if reasons:
        return False, "; ".join(reasons)
    return True, "single stage on a REAL worker: executable by `python -m partitioner execute` on that worker"
