"""Tensor serialization for the Phase 6 StageExecution protocol.

Encoding "raw-le-c-v1" (docs/REMOTE_EXECUTION.md):
  - elements in C (row-major) order;
  - each element little-endian: fp32 = IEEE-754 binary32, int64 = two's
    complement;
  - no header, no padding, no compression; shape and dtype travel in
    TensorPayload fields, never inside the bytes.
Deterministic: the same tensor always encodes to the same bytes.

Decoding never unpickles or evaluates anything. Every check that can be
made from the metadata (encoding, dtype, kind, rank, each dimension,
element count, byte length, size limit) runs BEFORE any tensor memory
is allocated; values are checked (finite) right after.
"""

from __future__ import annotations

from typing import Optional, Tuple

from partitioner.execution.interface import (
    HIDDEN_STATES,
    LOGITS,
    STAGE_INTERFACE_VERSION,
    TOKEN_IDS,
    Activation,
    ActivationMeta,
)

ENCODING = "raw-le-c-v1"
PROTOCOL_VERSION = 1

# dtype name (as in ActivationMeta) -> (proto enum name, numpy dtype, itemsize)
_DTYPES = {
    "fp32": ("TENSOR_DTYPE_FLOAT32", "<f4", 4),
    "int64": ("TENSOR_DTYPE_INT64", "<i8", 8),
}
_KINDS = {
    TOKEN_IDS: "ACTIVATION_KIND_TOKEN_IDS",
    HIDDEN_STATES: "ACTIVATION_KIND_HIDDEN_STATES",
    LOGITS: "ACTIVATION_KIND_LOGITS",
}
MAX_RANK = 3  # token ids are rank 2, hidden states and logits rank 3


class WireError(ValueError):
    """A tensor payload is invalid. `code` is the protocol error code."""

    def __init__(self, problems, code: str = "INVALID_TENSOR"):
        self.problems = list(problems)
        self.code = code
        super().__init__("; ".join(self.problems))


def pb2():
    from partitioner.workers import _ensure_worker_importable

    _ensure_worker_importable()
    from worker.generated import stage_execution_pb2

    return stage_execution_pb2


def payload_nbytes(shape, dtype: str) -> int:
    n = _DTYPES[dtype][2]
    for d in shape:
        n *= int(d)
    return n


def encode(activation: Activation):
    """Activation -> TensorPayload."""
    import numpy as np

    activation.check()
    m = activation.meta
    if m.dtype not in _DTYPES:
        raise WireError([f"dtype {m.dtype!r} is not supported by {ENCODING} (supported: {sorted(_DTYPES)})"])
    enum_name, np_dtype, _ = _DTYPES[m.dtype]
    arr = activation.tensor.detach().contiguous().cpu().numpy().astype(np_dtype, copy=False)
    p = pb2()
    return p.TensorPayload(
        encoding=ENCODING,
        dtype=p.TensorDType.Value(enum_name),
        shape=list(m.shape),
        data=arr.tobytes(order="C"),
        kind=p.ActivationKind.Value(_KINDS[m.kind]),
        produced_by_stage=m.produced_by_stage,
        positions_start=m.positions_start,
    )


def describe(payload) -> Tuple[Optional[str], Optional[str], Tuple[int, ...]]:
    """(kind, dtype, shape) from the metadata only; None for unknown enums."""
    p = pb2()
    kind_name = p.ActivationKind.Name(payload.kind) if payload.kind in p.ActivationKind.values() else None
    dtype_name = p.TensorDType.Name(payload.dtype) if payload.dtype in p.TensorDType.values() else None
    kind = next((k for k, v in _KINDS.items() if v == kind_name), None)
    dtype = next((d for d, v in _DTYPES.items() if v[0] == dtype_name), None)
    return kind, dtype, tuple(int(d) for d in payload.shape)


def check_metadata(payload, max_bytes: int) -> list:
    """Every check possible without touching the data bytes' values."""
    problems = []
    if payload.encoding != ENCODING:
        problems.append(f"encoding {payload.encoding!r} is not supported (supported: {ENCODING!r})")
    kind, dtype, shape = describe(payload)
    if dtype is None:
        problems.append(f"dtype enum {payload.dtype} is not supported (supported: {sorted(_DTYPES)})")
    if kind is None:
        problems.append(f"activation kind enum {payload.kind} is not supported")
    if not 1 <= len(shape) <= MAX_RANK:
        problems.append(f"rank {len(shape)} is not in 1..{MAX_RANK}")
    if any(d <= 0 for d in shape):
        problems.append(f"every dimension must be positive, got {list(shape)}")
    if payload.positions_start != 0:
        problems.append("positions_start must be 0 (single prefill pass only)")
    if problems or dtype is None:
        return problems
    expected = payload_nbytes(shape, dtype)  # Python ints: no overflow
    if expected > max_bytes:
        problems.append(f"tensor {list(shape)} {dtype} needs {expected} bytes > limit {max_bytes}")
    if len(payload.data) != expected:
        problems.append(f"payload has {len(payload.data)} bytes, shape {list(shape)} x {dtype} needs {expected}")
    return problems


def decode(payload, max_bytes: int, request_id: str = "") -> Activation:
    """TensorPayload -> Activation, or WireError listing every problem."""
    problems = check_metadata(payload, max_bytes)
    if problems:
        raise WireError(problems)
    import numpy as np
    import torch

    kind, dtype, shape = describe(payload)
    arr = np.frombuffer(payload.data, dtype=_DTYPES[dtype][1]).reshape(shape)
    tensor = torch.from_numpy(arr.astype(arr.dtype.newbyteorder("="), copy=True))  # owned, native-endian copy
    if dtype == "fp32" and not bool(torch.isfinite(tensor).all()):
        raise WireError(["tensor contains NaN or infinite values"])
    meta = ActivationMeta(STAGE_INTERFACE_VERSION, request_id, kind, shape, dtype, payload.produced_by_stage, payload.positions_start)
    return Activation(meta, tensor)
