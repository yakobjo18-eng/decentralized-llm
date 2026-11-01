"""Memory estimation: exact dtype arithmetic, quantization estimates,
KV cache, activations, tied-weight copies, provenance labels."""

from __future__ import annotations

import math

import pytest

from partitioner.memory import (
    DTYPE_BYTES,
    EstimationError,
    MemoryEstimator,
    MemoryPolicy,
    Precision,
    Workload,
    param_bytes,
)
from partitioner import model_spec as ms
from partitioner.provenance import Provenance

U_LM = -1  # index of lm_head (last unit)


def est(spec, **kw):
    precision = kw.pop("precision", Precision())
    workload = kw.pop("workload", Workload())
    policy = kw.pop("policy", MemoryPolicy())
    return MemoryEstimator(spec, precision, workload, policy)


def items(stage):
    return {i.name: i for i in stage.items}


def test_gpt2_fp32_weights_calculated_exactly(gpt2_spec):
    e = est(gpt2_spec)
    whole = e.stage(0, len(gpt2_spec.units))
    w = items(whole)["weights"]
    assert w.bytes == 124_439_808 * 4
    assert w.provenance == Provenance.CALCULATED
    assert "tied_weight_copy" not in items(whole)  # wte and lm_head together: one copy


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_half_precision_halves_weights(gpt2_spec, dtype):
    fp32 = est(gpt2_spec).stage(0, 15)
    half = est(gpt2_spec, precision=Precision(dtype)).stage(0, 15)
    assert items(half)["weights"].bytes * 2 == items(fp32)["weights"].bytes
    assert items(half)["kv_cache"].bytes * 2 == items(fp32)["kv_cache"].bytes


def test_gpt2_kv_cache_formula(gpt2_spec):
    for batch, ctx in [(1, 1024), (2, 512), (4, 128)]:
        e = est(gpt2_spec, workload=Workload(batch_size=batch, context_length=ctx))
        kv = items(e.stage(1, 13))["kv_cache"]
        assert kv.bytes == 12 * 2 * batch * ctx * 12 * 64 * 4
        assert kv.provenance == Provenance.CALCULATED


def test_int8_and_int4_estimates(gpt2_spec):
    c_fc = next(p for p in gpt2_spec.component("transformer.h.0").params if p.name.endswith("mlp.c_fc.weight"))
    assert c_fc.shape == (768, 3072)
    b8, p8, _ = param_bytes(c_fc, ms.TRANSFORMER_BLOCK, Precision("fp16", "int8"))
    assert b8 == 768 * 3072 + 3072 * 2 and p8 == Provenance.ESTIMATED
    b4, p4, _ = param_bytes(c_fc, ms.TRANSFORMER_BLOCK, Precision("fp16", "int4", quant_group_size=128))
    assert b4 == 768 * 3072 // 2 + math.ceil(768 * 3072 / 128) * 4 and p4 == Provenance.ESTIMATED
    # Non-linear parameters are not quantized: biases/norms stay fp16.
    bias = next(p for p in gpt2_spec.component("transformer.h.0").params if p.name.endswith("mlp.c_fc.bias"))
    assert param_bytes(bias, ms.TRANSFORMER_BLOCK, Precision("fp16", "int8"))[0] == 3072 * 2


def test_quantization_ordering_and_embeddings_untouched(gpt2_spec):
    totals = {q: items(est(gpt2_spec, precision=Precision("fp16", q)).stage(0, 15))["weights"].bytes
              for q in ("none", "int8", "int4")}
    assert totals["none"] > totals["int8"] > totals["int4"]
    # Embeddings are not quantized by default: unit 0 identical in all modes.
    e16 = est(gpt2_spec, precision=Precision("fp16")).units[0].owned_weight_bytes
    e8 = est(gpt2_spec, precision=Precision("fp16", "int8")).units[0].owned_weight_bytes
    assert e16 == e8 == (50257 + 1024) * 768 * 2
    eq = est(gpt2_spec, precision=Precision("fp16", "int8", quantize_embeddings=True)).units[0].owned_weight_bytes
    assert eq < e16


def test_quantized_plans_never_claim_a_runtime():
    assert Precision("fp16", "int8").to_dict()["quantization_runtime_implemented"] is False
    assert Precision("fp32").to_dict()["quantization_runtime_implemented"] is None


def test_tied_lm_head_copy_only_when_split(gpt2_spec):
    e = est(gpt2_spec)
    last = items(e.stage(13, 15))  # final_norm + lm_head, embeddings elsewhere
    assert last["tied_weight_copy"].bytes == 50257 * 768 * 4
    assert last["tied_weight_copy"].provenance == Provenance.CALCULATED
    assert "tied_weight_copy" not in items(e.stage(0, 15))
    assert "tied_weight_copy" not in items(e.stage(0, 5))


def test_activation_peak_is_max_not_sum(gpt2_spec):
    e = est(gpt2_spec)
    one = items(e.stage(1, 2))["activations_peak"].bytes
    many = items(e.stage(1, 13))["activations_peak"].bytes
    assert one == many  # activations are freed between layers
    assert items(e.stage(1, 2))["activations_peak"].provenance == Provenance.ESTIMATED


def test_gpt2_block_activation_formula(gpt2_spec):
    um = est(gpt2_spec).units[1]
    T = L = 1024
    hidden = T * 768
    assert um.activation_bytes == max(6 * hidden + 2 * 12 * T * L, 2 * hidden + 2 * T * 3072) * 4


def test_logits_last_shrinks_lm_head_activation(gpt2_spec):
    full = est(gpt2_spec).units[-1].activation_bytes
    last = est(gpt2_spec, workload=Workload(logits_positions="last")).units[-1].activation_bytes
    assert full == (1024 * 768 + 1024 * 50257) * 4 and last == (1024 * 768 + 50257) * 4


def test_runtime_overhead_is_labelled_measured_and_configurable(gpt2_spec):
    it = items(est(gpt2_spec).stage(0, 15))
    assert it["runtime_overhead"].provenance == Provenance.MEASURED
    assert it["runtime_overhead"].bytes == 700 * 2**20
    custom = items(est(gpt2_spec, policy=MemoryPolicy(runtime_overhead_bytes=123,
                                                      runtime_overhead_provenance=Provenance.ESTIMATED)).stage(0, 15))
    assert custom["runtime_overhead"].bytes == 123 and custom["runtime_overhead"].provenance == Provenance.ESTIMATED


def test_stage_totals_and_provenance(gpt2_spec):
    s = est(gpt2_spec).stage(0, 15)
    assert s.total_bytes == sum(i.bytes for i in s.items)
    assert s.host_bytes == items(s)["runtime_overhead"].bytes + items(s)["load_temp_buffer"].bytes
    assert s.provenance == Provenance.ESTIMATED  # weakest part decides
    assert "outgoing_hidden_state_buffer" not in items(s)  # last stage sends nothing
    assert "outgoing_hidden_state_buffer" in items(est(gpt2_spec).stage(0, 5))


def test_kv_and_load_temp_can_be_excluded(gpt2_spec):
    s = items(est(gpt2_spec, policy=MemoryPolicy(include_kv_cache=False, include_load_temp=False)).stage(0, 15))
    assert "kv_cache" not in s and "load_temp_buffer" not in s


def test_scales_with_synthetic_model(small_spec):
    e = est(small_spec, workload=Workload(batch_size=2, context_length=64))
    kv = items(e.stage(1, 5))["kv_cache"].bytes
    assert kv == 4 * 2 * 2 * 64 * 4 * 16 * 4  # 4 blocks, 4 heads x head_dim 16


@pytest.mark.parametrize(
    "precision, workload, policy, message",
    [
        (Precision("fp8"), Workload(), MemoryPolicy(), "weight_dtype"),
        (Precision("fp16", "int3"), Workload(), MemoryPolicy(), "quantization"),
        (Precision(activation_dtype="int8"), Workload(), MemoryPolicy(), "activation_dtype"),
        (Precision(), Workload(context_length=4096), MemoryPolicy(), "context_length"),
        (Precision(), Workload(context_length=128, prompt_length=256), MemoryPolicy(), "prompt_length"),
        (Precision(), Workload(batch_size=0), MemoryPolicy(), "batch_size"),
        (Precision(), Workload(cost_mode="train"), MemoryPolicy(), "cost_mode"),
        (Precision(), Workload(), MemoryPolicy(extra_safety_fraction=1.5), "extra_safety_fraction"),
    ],
)
def test_invalid_settings_rejected(gpt2_spec, precision, workload, policy, message):
    with pytest.raises(EstimationError, match=message):
        MemoryEstimator(gpt2_spec, precision, workload, policy)


def test_invalid_ranges_rejected(gpt2_spec):
    e = est(gpt2_spec)
    for bad in [(3, 3), (5, 2), (-1, 4), (0, 16)]:
        with pytest.raises(EstimationError):
            e.stage(*bad)


def test_dtype_table():
    assert DTYPE_BYTES == {"fp32": 4, "fp16": 2, "bf16": 2}
