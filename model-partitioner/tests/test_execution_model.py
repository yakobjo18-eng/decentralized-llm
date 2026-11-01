"""The built-in tiny model: structure, deterministic weights, known
output, independent oracles (NumPy float64, Hugging Face), partitioned
vs. unpartitioned equality, and stage range / tensor validation."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
import torch

from partitioner.execution import tiny_gpt2 as t
from partitioner.execution.interface import (
    HIDDEN_STATES,
    INPUT_STAGE,
    STAGE_INTERFACE_VERSION,
    TOKEN_IDS,
    Activation,
    ActivationMeta,
    InProcessTransport,
    InterfaceError,
)
from partitioner.execution.runner import REFERENCE_ATOL

IDS = [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]]
# fp32 implementation vs. a float64 oracle / Hugging Face's kernels:
# observed max |diff| 1.2e-4 (NumPy) and 2.0e-4 (HF) on logits of
# magnitude ~11, i.e. ~1e-5 relative -- fp32 rounding.
ORACLE_ATOL = 5e-4


def np_forward(w, ids, n_layer=4, n_head=4, eps=1e-5):
    """Independent GPT-2 forward pass in float64 NumPy (no torch ops)."""
    W = {k: v.double().numpy() for k, v in w.items()}

    def ln(x, g, b):
        mu = x.mean(-1, keepdims=True)
        var = ((x - mu) ** 2).mean(-1, keepdims=True)
        return (x - mu) / np.sqrt(var + eps) * g + b

    def gelu(x):
        return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))

    B, T = ids.shape
    x = W["transformer.wte.weight"][ids] + W["transformer.wpe.weight"][:T]
    H = x.shape[-1]
    hd = H // n_head
    for i in range(n_layer):
        p = f"transformer.h.{i}."
        h = ln(x, W[p + "ln_1.weight"], W[p + "ln_1.bias"])
        qkv = h @ W[p + "attn.c_attn.weight"] + W[p + "attn.c_attn.bias"]
        q, k, v = [a.reshape(B, T, n_head, hd).transpose(0, 2, 1, 3) for a in np.split(qkv, 3, axis=-1)]
        s = q @ k.transpose(0, 1, 3, 2) / math.sqrt(hd)
        s = np.where(np.tril(np.ones((T, T), bool)), s, -np.inf)
        s = np.exp(s - s.max(-1, keepdims=True))
        s /= s.sum(-1, keepdims=True)
        y = (s @ v).transpose(0, 2, 1, 3).reshape(B, T, H)
        x = x + y @ W[p + "attn.c_proj.weight"] + W[p + "attn.c_proj.bias"]
        h = ln(x, W[p + "ln_2.weight"], W[p + "ln_2.bias"])
        x = x + gelu(h @ W[p + "mlp.c_fc.weight"] + W[p + "mlp.c_fc.bias"]) @ W[p + "mlp.c_proj.weight"] + W[p + "mlp.c_proj.bias"]
    x = ln(x, W["transformer.ln_f.weight"], W["transformer.ln_f.bias"])
    return x @ W["lm_head.weight"].T


def run_chain(spec, ranges, ids):
    stages = [t.TinyGPT2Stage(t.make_stage_spec(spec, k, len(ranges), s, e), spec) for k, (s, e) in enumerate(ranges)]
    transport = InProcessTransport()
    x = Activation(ActivationMeta(STAGE_INTERFACE_VERSION, "r", TOKEN_IDS, tuple(ids.shape), "int64", INPUT_STAGE), ids)
    for k, st in enumerate(stages):
        x = st.forward(x)
        if k + 1 < len(stages):
            x = transport.send(x, k + 1)
    return x, transport


# ---- structure and weights ---------------------------------------------


def test_spec_comes_from_the_phase4_gpt2_adapter(tiny_spec):
    h, inner, V, P, L = 32, 128, 128, 32, 4
    by_hand = V * h + P * h + L * (4 * h * h + 2 * h * inner + 9 * h + inner) + 2 * h
    assert tiny_spec.total_params == by_hand == 56_000
    assert [u.name for u in tiny_spec.units] == ["embeddings", "block.0", "block.1", "block.2", "block.3", "final_norm", "lm_head"]
    assert tiny_spec.model_id == t.MODEL_ID and tiny_spec.adapter == "gpt2"
    assert ("lm_head.weight", "transformer.wte.weight") in tiny_spec.tied_weights


def test_weights_are_deterministic_and_cover_exactly_the_model(tiny_spec):
    a, b = t.full_weights(tiny_spec), t.full_weights(tiny_spec)
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    expected = {p.name for c in tiny_spec.components for p in c.params}
    assert set(a) == expected
    for c in tiny_spec.components:
        for p in c.params:
            assert tuple(a[p.name].shape) == p.shape and a[p.name].dtype == torch.float32
    # Tied on one stage: the SAME tensor, not a copy.
    assert a["lm_head.weight"].data_ptr() == a["transformer.wte.weight"].data_ptr()


def test_stage_holds_only_its_own_components(tiny_spec):
    w = t.stage_weights(tiny_spec, 1, 3)  # block.0, block.1
    assert w and all(k.startswith(("transformer.h.0.", "transformer.h.1.")) for k in w)
    last = t.stage_weights(tiny_spec, 5, 7)  # final_norm + lm_head, embeddings elsewhere
    assert set(last) == {"transformer.ln_f.weight", "transformer.ln_f.bias", "lm_head.weight"}
    assert torch.equal(last["lm_head.weight"], t.full_weights(tiny_spec)["transformer.wte.weight"])  # its own copy
    assert t.weight_bytes(tiny_spec, 0, 7) == tiny_spec.total_params * 4
    assert t.weight_bytes(tiny_spec, 5, 7) == (2 * 32 + 128 * 32) * 4


# ---- known output and independent oracles --------------------------------


def test_known_expected_output():
    """Golden values for fixed weights (formula-sin-v1) and fixed input.
    They come from the independent float64 NumPy forward pass above,
    rounded to 6 decimals -- not from the implementation under test."""
    logits = t.reference_forward(torch.tensor(IDS))
    assert logits.shape == (2, 8, 128) and logits.dtype == torch.float32
    expected_row = [-6.123801, -1.536287, -0.31588, -3.619174, -7.15987, -5.839801]
    assert np.allclose(logits[0, -1, :6].numpy(), expected_row, atol=ORACLE_ATOL)
    assert np.allclose(logits[1, 3, 100:104].numpy(), [0.235758, -3.501052, -3.373231, 2.391517], atol=ORACLE_ATOL)
    assert logits[:, -1].argmax(-1).tolist() == [60, 96]
    assert torch.equal(logits, t.reference_forward(torch.tensor(IDS)))  # run-to-run deterministic


def test_matches_independent_numpy_float64_implementation(tiny_spec):
    ids = np.array(IDS)
    ref = np_forward(t.full_weights(tiny_spec), ids)
    mine = t.reference_forward(torch.tensor(ids)).double().numpy()
    assert np.abs(mine - ref).max() <= ORACLE_ATOL


def test_matches_huggingface_gpt2_with_the_same_weights(tiny_spec):
    from transformers import GPT2LMHeadModel

    hf = GPT2LMHeadModel(t.config()).eval()
    hf.load_state_dict(t.full_weights(tiny_spec), strict=True)
    with torch.no_grad():
        hf_logits = hf(torch.tensor(IDS)).logits
    assert (hf_logits - t.reference_forward(torch.tensor(IDS))).abs().max().item() <= ORACLE_ATOL


def test_attention_is_causal():
    a = t.reference_forward(torch.tensor([[5, 6, 7, 8]]))
    b = t.reference_forward(torch.tensor([[5, 6, 7, 99]]))
    assert torch.allclose(a[:, :3], b[:, :3], atol=1e-6, rtol=0)
    assert not torch.allclose(a[:, 3], b[:, 3])


# ---- partitioned vs. unpartitioned --------------------------------------


@pytest.mark.parametrize("ranges", [
    [(0, 7)],
    *[[(0, k), (k, 7)] for k in range(1, 7)],
    [(0, 2), (2, 4), (4, 7)],
    [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)],
])
def test_partitioned_matches_unpartitioned_reference(tiny_spec, ranges):
    ids = torch.tensor(IDS)
    out, transport = run_chain(tiny_spec, ranges, ids)
    ref = t.reference_forward(ids)
    assert out.meta.kind == "logits" and out.tensor.shape == ref.shape and out.tensor.dtype == ref.dtype
    assert (out.tensor - ref).abs().max().item() <= REFERENCE_ATOL
    assert len(transport.handoffs) == len(ranges) - 1
    for h in transport.handoffs:
        assert h.kind == HIDDEN_STATES and h.shape == (2, 8, 32) and h.nbytes == 2 * 8 * 32 * 4


# ---- stage range and tensor validation ----------------------------------


@pytest.mark.parametrize("change, message", [
    (dict(unit_range=(3, 3)), "non-empty range"),
    (dict(unit_range=(4, 2)), "non-empty range"),
    (dict(unit_range=(0, 8)), "non-empty range"),
    (dict(unit_names=("block.0",)), "unit_names"),
    (dict(components=("transformer.h.0",)), "components"),
    (dict(model_id="gpt2"), "is for model"),
    (dict(config_fingerprint="0" * 64), "config_fingerprint"),
    (dict(activation_dtype="fp16"), "not executable"),
    (dict(interface_version=2), "interface_version"),
    (dict(input_kind=TOKEN_IDS), "input_kind"),
    (dict(output_kind="logits"), "output_kind"),
])
def test_stage_spec_is_checked_against_the_model(tiny_spec, change, message):
    good = t.make_stage_spec(tiny_spec, 1, 3, 1, 3)
    t.TinyGPT2Stage(good, tiny_spec)  # sanity: the unmodified spec is accepted
    with pytest.raises(InterfaceError, match=message):
        t.TinyGPT2Stage(dataclasses.replace(good, **change), tiny_spec)


def _hidden(shape=(1, 4, 32), dtype=torch.float32, kind=HIDDEN_STATES, producer=0, meta_dtype="fp32", meta_shape=None):
    x = torch.zeros(shape, dtype=dtype)
    return Activation(ActivationMeta(STAGE_INTERFACE_VERSION, "r", kind, meta_shape or shape, meta_dtype, producer), x)


@pytest.mark.parametrize("activation, message", [
    (_hidden(kind=TOKEN_IDS), "expected a hidden_states activation"),
    (_hidden(producer=INPUT_STAGE), "comes from stage -1"),
    (_hidden(shape=(1, 4, 16)), "shape"),
    (_hidden(dtype=torch.float64, meta_dtype="float64"), "dtype float64 != expected fp32"),
    (_hidden(shape=(1, 33, 32)), "33 tokens"),
    (_hidden(meta_shape=(1, 5, 32)), "tensor shape"),  # metadata doesn't describe the tensor
    (_hidden(meta_dtype="bf16"), "tensor dtype"),
])
def test_stage_rejects_wrong_inputs(tiny_spec, activation, message):
    stage = t.TinyGPT2Stage(t.make_stage_spec(tiny_spec, 1, 2, 1, 3), tiny_spec)
    with pytest.raises(InterfaceError, match=message):
        stage.forward(activation)


def test_first_stage_rejects_out_of_vocabulary_tokens(tiny_spec):
    stage = t.TinyGPT2Stage(t.make_stage_spec(tiny_spec, 0, 1, 0, 7), tiny_spec)
    ids = torch.tensor([[1, 128]])
    with pytest.raises(InterfaceError, match=r"token ids must be in \[0, 128\)"):
        stage.forward(Activation(ActivationMeta(STAGE_INTERFACE_VERSION, "r", TOKEN_IDS, (1, 2), "int64", INPUT_STAGE), ids))


def test_transport_checks_metadata_before_handing_over():
    bad = _hidden(meta_shape=(9, 9, 9))
    with pytest.raises(InterfaceError, match="tensor shape"):
        InProcessTransport().send(bad, 1)
