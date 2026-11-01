"""Resource limits, checked BEFORE any tensor is allocated.

The input is validated as plain Python lists; sizes are computed from
the model spec (CALCULATED boundary tensors) and the Phase 4 activation
estimator (ESTIMATED per-unit peaks, eager attention). Only if every
check passes does the runner create the token tensor and the weights.
"""

from __future__ import annotations

import dataclasses
from typing import Any, List, Optional, Sequence

from partitioner import model_spec as ms
from partitioner.memory import DTYPE_BYTES, Precision, Workload, unit_memory
from partitioner.plan import fmt_bytes

MIB = 2**20


@dataclasses.dataclass(frozen=True)
class ExecutionLimits:
    max_batch_size: int = 8
    # None = the model's maximum positions.
    max_sequence_length: Optional[int] = None
    # Largest single tensor or per-unit activation peak allowed.
    max_activation_bytes: int = 64 * MIB
    # Total weight bytes all stages of one run may materialize.
    max_weight_bytes: int = 64 * MIB

    def validate(self) -> List[str]:
        p = []
        if self.max_batch_size <= 0:
            p.append("max_batch_size must be positive")
        if self.max_sequence_length is not None and self.max_sequence_length <= 0:
            p.append("max_sequence_length must be positive")
        if self.max_activation_bytes <= 0:
            p.append("max_activation_bytes must be positive")
        if self.max_weight_bytes <= 0:
            p.append("max_weight_bytes must be positive")
        return p

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def check_token_ids(token_ids: Any, vocab_size: int) -> List[str]:
    """Shape and value checks on the raw input, without building a tensor."""
    if not isinstance(token_ids, (list, tuple)) or not token_ids:
        return ["token_ids must be a non-empty list of sequences"]
    p = []
    lengths = set()
    for b, row in enumerate(token_ids):
        if not isinstance(row, (list, tuple)) or not row:
            p.append(f"sequence {b} must be a non-empty list of token ids")
            continue
        lengths.add(len(row))
        bad = [t for t in row if isinstance(t, bool) or not isinstance(t, int) or not 0 <= t < vocab_size]
        if bad:
            p.append(f"sequence {b}: token ids must be integers in [0, {vocab_size}); got {bad[:5]}")
    if len(lengths) > 1:
        p.append(f"all sequences must have the same length (no padding/attention mask support); got lengths {sorted(lengths)}")
    return p


@dataclasses.dataclass(frozen=True)
class Preflight:
    batch: int
    tokens: int
    largest_boundary_bytes: int
    largest_unit_activation_bytes: int
    weight_bytes: int

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["boundary_bytes_provenance"] = "CALCULATED"
        d["unit_activation_provenance"] = "ESTIMATED (Phase 4 activation model, eager attention)"
        return d


def preflight(spec: ms.ModelSpec, batch: int, tokens: int, stage_ranges: Sequence[Sequence[int]],
              activation_dtype: str, weight_bytes: int, limits: ExecutionLimits,
              plan_batch: Optional[int] = None, plan_prompt: Optional[int] = None):
    """Returns (Preflight or None, problems). Allocates nothing."""
    p = limits.validate()
    max_t = min(spec.max_positions, limits.max_sequence_length or spec.max_positions)
    if batch > limits.max_batch_size:
        p.append(f"batch size {batch} > limit max_batch_size {limits.max_batch_size}")
    if tokens > max_t:
        p.append(f"sequence length {tokens} > limit {max_t} (model positions {spec.max_positions}, "
                 f"max_sequence_length {limits.max_sequence_length})")
    if plan_batch is not None and batch > plan_batch:
        p.append(f"batch size {batch} > the plan's workload batch_size {plan_batch} (memory was planned for that)")
    if plan_prompt is not None and tokens > plan_prompt:
        p.append(f"sequence length {tokens} > the plan's workload prompt_length {plan_prompt} (memory was planned for that)")
    if weight_bytes > limits.max_weight_bytes:
        p.append(f"weights {fmt_bytes(weight_bytes)} > limit max_weight_bytes {fmt_bytes(limits.max_weight_bytes)}")
    if p:
        return None, p

    a = DTYPE_BYTES[activation_dtype]
    boundaries = [batch * tokens * 8]  # int64 token ids
    for _, end in stage_ranges:
        width = spec.vocab_size if spec.units[end - 1].kind == ms.LM_HEAD else spec.hidden_size
        boundaries.append(batch * tokens * width * a)
    wl = Workload(batch_size=batch, context_length=tokens, prompt_length=tokens, logits_positions="all").resolve(spec)
    precision = Precision(weight_dtype=activation_dtype)
    unit_peak = max(unit_memory(spec, u, precision, wl).activation_bytes for u in spec.units)
    largest = max(boundaries)
    if largest > limits.max_activation_bytes:
        p.append(f"largest boundary tensor {fmt_bytes(largest)} > limit max_activation_bytes {fmt_bytes(limits.max_activation_bytes)}")
    if unit_peak > limits.max_activation_bytes:
        p.append(f"estimated per-unit activation peak {fmt_bytes(unit_peak)} > limit max_activation_bytes "
                 f"{fmt_bytes(limits.max_activation_bytes)}")
    if p:
        return None, p
    return Preflight(batch, tokens, largest, unit_peak, weight_bytes), []
