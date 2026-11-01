"""Phase 6 tensor serialization (raw-le-c-v1): round trips, exact byte
layout, and rejection of malformed payloads before allocation."""

from __future__ import annotations

import pickle

import pytest
import torch

from partitioner.execution import wire
from partitioner.execution.interface import STAGE_INTERFACE_VERSION, Activation, ActivationMeta

pb = wire.pb2()
LIMIT = 1 << 20


def act(tensor, kind="hidden_states", dtype="fp32", producer=0):
    return Activation(ActivationMeta(STAGE_INTERFACE_VERSION, "r", kind, tuple(tensor.shape), dtype, producer), tensor)


def payload(**kw):
    base = dict(encoding=wire.ENCODING, dtype=pb.TENSOR_DTYPE_FLOAT32, shape=[1, 2, 2], data=b"\x00" * 16,
                kind=pb.ACTIVATION_KIND_HIDDEN_STATES, produced_by_stage=0)
    base.update(kw)
    return pb.TensorPayload(**base)


@pytest.mark.parametrize("tensor, kind, dtype", [
    (torch.randn(2, 8, 32), "hidden_states", "fp32"),
    (torch.randn(1, 3, 128), "logits", "fp32"),
    (torch.tensor([[1, 2, 3], [127, 0, 5]], dtype=torch.int64), "token_ids", "int64"),
])
def test_round_trip_is_bit_exact(tensor, kind, dtype):
    p = wire.encode(act(tensor, kind, dtype, producer=-1 if kind == "token_ids" else 1))
    assert p.encoding == "raw-le-c-v1" and list(p.shape) == list(tensor.shape)
    assert len(p.data) == tensor.numel() * tensor.element_size()
    back = wire.decode(pb.TensorPayload.FromString(p.SerializeToString()), LIMIT, "r")
    assert torch.equal(back.tensor, tensor) and back.tensor.dtype == tensor.dtype
    assert (back.meta.kind, back.meta.dtype, back.meta.shape) == (kind, dtype, tuple(tensor.shape))
    assert back.meta.produced_by_stage == p.produced_by_stage
    assert back.tensor.data_ptr() != 0 and back.tensor.is_contiguous()


def test_byte_layout_is_little_endian_row_major_and_deterministic():
    t = torch.tensor([[[1.0, -2.0]]])
    p = wire.encode(act(t))
    assert p.data == b"\x00\x00\x80\x3f" + b"\x00\x00\x00\xc0"  # 1.0, -2.0 as LE binary32
    ids = wire.encode(act(torch.tensor([[1, 256]]), "token_ids", "int64", -1))
    assert ids.data == (1).to_bytes(8, "little") + (256).to_bytes(8, "little")
    assert wire.encode(act(t)).SerializeToString() == p.SerializeToString()
    # A non-contiguous view encodes in logical (C) order.
    base = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)
    assert wire.encode(act(base.transpose(1, 2).contiguous())).data == wire.encode(act(base.transpose(1, 2))).data


@pytest.mark.parametrize("kw, message", [
    (dict(encoding="pickle"), "encoding 'pickle' is not supported"),
    (dict(encoding=""), "encoding '' is not supported"),
    (dict(dtype=pb.TENSOR_DTYPE_UNSPECIFIED), "dtype enum 0 is not supported"),
    (dict(dtype=99), "dtype enum 99 is not supported"),
    (dict(kind=pb.ACTIVATION_KIND_UNSPECIFIED), "activation kind enum 0"),
    (dict(shape=[1, 0, 2], data=b""), "every dimension must be positive"),
    (dict(shape=[1, -2, 2]), "every dimension must be positive"),
    (dict(shape=[], data=b""), "rank 0"),
    (dict(shape=[1, 1, 2, 2]), "rank 4"),
    (dict(data=b"\x00" * 15), "payload has 15 bytes"),
    (dict(data=b"\x00" * 17), "payload has 17 bytes"),
    (dict(shape=[1, 2, 3]), "needs 24"),
    (dict(positions_start=4), "positions_start must be 0"),
])
def test_malformed_payloads_are_rejected(kw, message):
    with pytest.raises(wire.WireError) as exc:
        wire.decode(payload(**kw), LIMIT)
    assert any(message in p for p in exc.value.problems), exc.value.problems
    assert exc.value.code == "INVALID_TENSOR"


def test_oversized_shape_is_rejected_from_metadata_alone(monkeypatch):
    """A huge declared shape with a tiny body: refused before numpy or
    torch touch the data."""
    import numpy as np

    monkeypatch.setattr(np, "frombuffer", lambda *a, **k: pytest.fail("decoded despite the size check"))
    p = payload(shape=[10**6, 10**6, 10**3], data=b"\x00" * 16)
    with pytest.raises(wire.WireError) as exc:
        wire.decode(p, LIMIT)
    assert any("> limit" in m for m in exc.value.problems)


def test_size_limit_is_enforced_exactly():
    p = payload()  # 16 bytes
    assert wire.decode(p, 16).tensor.shape == (1, 2, 2)
    with pytest.raises(wire.WireError, match="needs 16 bytes > limit 15"):
        wire.decode(p, 15)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_values_are_rejected(value):
    t = torch.zeros(1, 2, 2)
    t[0, 1, 0] = value
    with pytest.raises(wire.WireError, match="NaN or infinite"):
        wire.decode(wire.encode(act(t)), LIMIT)


def test_a_pickle_body_is_just_bytes():
    """The decoder never unpickles: a pickle stream is either rejected by
    the length check or read as plain numbers."""
    body = pickle.dumps({"__reduce__": "os.system"})
    with pytest.raises(wire.WireError, match="payload has"):
        wire.decode(payload(data=body), LIMIT)


def test_encode_refuses_unsupported_dtypes():
    with pytest.raises(wire.WireError, match="not supported"):
        wire.encode(act(torch.zeros(1, 2, 2, dtype=torch.bfloat16), dtype="bf16"))
