"""Versioned execution-stage interface (STAGE_INTERFACE_VERSION 1).

The narrow contract between pipeline stages, designed so a later phase
can put a network between two stages without changing either side:

  StageSpec            what one stage runs: a contiguous [start, end)
                       range of plan units, plus the model identity it
                       was built for. Derived from a plan stage.
  Activation           what crosses a stage boundary: one tensor plus
                       explicit metadata (kind, shape, dtype, request id,
                       producing stage). Nothing is inferred on the
                       receiving side.
  StageExecutor        runs one StageSpec: Activation in, Activation out.
  ActivationTransport  moves an Activation from stage k to stage k+1.

Only InProcessTransport exists in Phase 5: it hands the same Python
object to the next stage, in the same process. There is no
serialization, no gRPC message and no remote transfer. A future remote
transport must carry every ActivationMeta field (they are all plain
ints/strings) next to the raw tensor bytes.
"""

from __future__ import annotations

import abc
import dataclasses
from typing import Any, List, Optional, Tuple

STAGE_INTERFACE_VERSION = 1

# Activation kinds. The first stage consumes TOKEN_IDS; a stage that
# ends with the LM head produces LOGITS; every other boundary carries
# HIDDEN_STATES of shape [batch, tokens, hidden_size].
TOKEN_IDS = "token_ids"
HIDDEN_STATES = "hidden_states"
LOGITS = "logits"
INPUT_STAGE = -1  # produced_by_stage for the request's own input


class InterfaceError(ValueError):
    """An activation or stage violates the interface contract (wrong
    version, kind, shape, dtype, or a range that doesn't fit the model)."""


@dataclasses.dataclass(frozen=True)
class StageSpec:
    interface_version: int
    model_id: str
    config_fingerprint: str
    stage_index: int
    n_stages: int
    unit_range: Tuple[int, int]  # [start, end) in plan unit indices
    unit_names: Tuple[str, ...]
    components: Tuple[str, ...]
    input_kind: str
    output_kind: str
    activation_dtype: str  # "fp32" etc., as in the plan's precision

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["unit_range"] = list(self.unit_range)
        d["unit_names"] = list(self.unit_names)
        d["components"] = list(self.components)
        return d


@dataclasses.dataclass(frozen=True)
class ActivationMeta:
    interface_version: int
    request_id: str
    kind: str
    shape: Tuple[int, ...]
    dtype: str  # "int64" for token ids, else the plan's activation dtype
    produced_by_stage: int
    # First position of these tokens in the sequence. Always 0 in
    # Phase 5 (single prefill pass, no KV cache); carried explicitly so
    # a later decode step never re-derives positions on a worker.
    positions_start: int = 0

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["shape"] = list(self.shape)
        return d


@dataclasses.dataclass(frozen=True)
class Activation:
    meta: ActivationMeta
    tensor: Any  # torch.Tensor; typed loosely so this module needs no torch import

    def check(self) -> None:
        """The tensor must match its metadata exactly."""
        from partitioner.execution.dtypes import torch_dtype_name

        problems = []
        if self.meta.interface_version != STAGE_INTERFACE_VERSION:
            problems.append(
                f"interface_version {self.meta.interface_version} != supported {STAGE_INTERFACE_VERSION}"
            )
        if tuple(self.tensor.shape) != tuple(self.meta.shape):
            problems.append(f"tensor shape {tuple(self.tensor.shape)} != declared {tuple(self.meta.shape)}")
        actual = torch_dtype_name(self.tensor.dtype)
        if actual != self.meta.dtype:
            problems.append(f"tensor dtype {actual} != declared {self.meta.dtype}")
        if problems:
            raise InterfaceError(f"{self.meta.kind} activation from stage {self.meta.produced_by_stage}: " + "; ".join(problems))


class StageExecutor(abc.ABC):
    """Runs one stage. Implementations hold ONLY the weights of their
    own components (plus a copy of any tied weight whose owner lives on
    another stage) and must reject activations they weren't built for."""

    spec: StageSpec

    @abc.abstractmethod
    def forward(self, activation: Activation) -> Activation:
        """Run this stage's units in order on `activation`."""


@dataclasses.dataclass(frozen=True)
class Handoff:
    """Record of one boundary crossing, for the execution result."""

    from_stage: int
    to_stage: int
    kind: str
    shape: Tuple[int, ...]
    dtype: str
    nbytes: int
    transport: str


class ActivationTransport(abc.ABC):
    name: str = ""
    # True only for a transport that actually leaves the process. No
    # such transport exists in Phase 5.
    crosses_process_boundary: bool = False

    def __init__(self) -> None:
        self.handoffs: List[Handoff] = []

    @abc.abstractmethod
    def send(self, activation: Activation, to_stage: int) -> Activation:
        """Deliver `activation` to stage `to_stage`; returns what that
        stage receives."""


class InProcessTransport(ActivationTransport):
    """Local only: the receiving stage gets the very same tensor object.
    No copy, no serialization, no network."""

    name = "in-process (no serialization, no network)"
    crosses_process_boundary = False

    def send(self, activation: Activation, to_stage: int) -> Activation:
        activation.check()
        t = activation.tensor
        self.handoffs.append(Handoff(
            activation.meta.produced_by_stage, to_stage, activation.meta.kind, tuple(t.shape),
            activation.meta.dtype, t.element_size() * t.numel(), self.name,
        ))
        return activation


def expected_output(kind: str, batch: int, tokens: int, hidden_size: int, vocab_size: int) -> Optional[Tuple[int, ...]]:
    if kind == HIDDEN_STATES:
        return (batch, tokens, hidden_size)
    if kind == LOGITS:
        return (batch, tokens, vocab_size)
    if kind == TOKEN_IDS:
        return (batch, tokens)
    return None
