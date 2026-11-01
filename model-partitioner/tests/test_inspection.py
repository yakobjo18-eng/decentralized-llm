"""Architecture inspection: the real GPT-2 (ground truth), synthetic
configs (no hardcoded layer count), and rejection of invalid or
unsupported models."""

from __future__ import annotations

import pytest

from partitioner import model_spec as ms
from partitioner.adapters import AdapterError, GPT2Adapter, get_adapter
from partitioner.inspection import ModelInspectionError, inspect_model


def test_real_gpt2_architecture(gpt2_spec):
    s = gpt2_spec
    assert s.architecture == "gpt2" and s.adapter == "gpt2"
    assert (s.n_layers, s.hidden_size, s.attention.n_heads, s.attention.head_dim) == (12, 768, 12, 64)
    assert (s.vocab_size, s.max_positions, s.intermediate_size) == (50257, 1024, 3072)
    assert s.config["activation_function"] == "gelu_new"


def test_real_gpt2_parameter_count_matches_independent_formula(gpt2_spec):
    """124,439,808 from the real model, and from GPT-2's published
    layout computed by hand: wte + wpe + L x (12h^2 + 13h) + ln_f."""
    V, P, h, L = 50257, 1024, 768, 12
    by_hand = V * h + P * h + L * (12 * h * h + 13 * h) + 2 * h
    assert gpt2_spec.total_params == by_hand == 124_439_808
    owned = sum(c.owned_numel for c in gpt2_spec.components)
    assert owned == gpt2_spec.total_params  # tied lm_head counted once


def test_real_gpt2_components_and_units(gpt2_spec):
    s = gpt2_spec
    assert [u.name for u in s.units] == ["embeddings"] + [f"block.{i}" for i in range(12)] + ["final_norm", "lm_head"]
    assert [u.index for u in s.units] == list(range(15))
    kinds = {c.name: c.kind for c in s.components}
    assert kinds["transformer.wte"] == ms.EMBEDDING and kinds["transformer.wpe"] == ms.POSITIONAL_EMBEDDING
    assert kinds["transformer.ln_f"] == ms.FINAL_NORM and kinds["lm_head"] == ms.LM_HEAD
    assert sum(1 for c in s.components if c.kind == ms.TRANSFORMER_BLOCK) == 12
    # Every component in exactly one unit.
    in_units = [c for u in s.units for c in u.components]
    assert sorted(in_units) == sorted(c.name for c in s.components) and len(in_units) == len(set(in_units))


def test_real_gpt2_tied_lm_head_detected_by_identity(gpt2_spec):
    assert gpt2_spec.tied_weights == (("lm_head.weight", "transformer.wte.weight"),)
    lm = gpt2_spec.component("lm_head")
    assert lm.owned_numel == 0 and lm.tied_numel == 50257 * 768


def test_real_gpt2_inspection_loaded_no_weights(gpt2_spec):
    info = gpt2_spec.inspection
    assert "no weights loaded" in info["method"]
    # Building the meta model costs a few MiB, not the ~475 MiB the fp32 weights would.
    assert info["rss_after_mib"] - info["rss_before_mib"] < 100


def test_real_gpt2_checkpoint_header_cross_check(gpt2_spec):
    ck = gpt2_spec.checkpoint
    if ck is None or ck.status == "unavailable":
        pytest.skip(f"checkpoint header not reachable: {ck.detail if ck else 'not checked'}")
    assert ck.status == "verified", ck.detail
    assert ck.stored_dtypes == ("F32",)
    assert not ck.missing_params and not ck.shape_mismatches
    # The published file also stores 12 legacy causal-mask buffers
    # (h.N.attn.bias, 1x1x1024x1024) that are not parameters.
    assert len(ck.extra_tensors) == 12 and all(t.endswith(".attn.bias") for t in ck.extra_tensors)
    assert ck.stored_values == gpt2_spec.total_params + 12 * 1024 * 1024


def test_synthetic_config_layer_count_not_hardcoded(small_spec):
    assert small_spec.n_layers == 4
    assert len(small_spec.units) == 4 + 3
    assert small_spec.total_params == 1000 * 64 + 128 * 64 + 4 * (12 * 64 * 64 + 13 * 64) + 2 * 64
    assert small_spec.checkpoint is None


def test_custom_intermediate_size(small_spec):
    from transformers import GPT2Config

    spec = inspect_model(config=GPT2Config(n_layer=2, n_embd=64, n_head=4, n_inner=100, vocab_size=50, n_positions=16),
                         check_checkpoint=False)
    assert spec.intermediate_size == 100
    fc = next(p for p in spec.component("transformer.h.0").params if p.name.endswith("mlp.c_fc.weight"))
    assert fc.shape == (64, 100)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(n_embd=100, n_head=3), "divisible"),
        (dict(n_layer=0), "n_layer must be a positive integer"),
        (dict(vocab_size=-5), "vocab_size must be a positive integer"),
        (dict(n_inner=0), "n_inner"),
        (dict(add_cross_attention=True), "cross-attention"),
    ],
)
def test_invalid_gpt2_configs_rejected(kwargs, message):
    from transformers import GPT2Config

    base = dict(n_layer=2, n_embd=64, n_head=4, vocab_size=100, n_positions=16)
    base.update(kwargs)
    cfg = GPT2Config(**base)
    with pytest.raises(AdapterError, match=message):
        inspect_model(config=cfg, check_checkpoint=False)


def test_unsupported_architecture_rejected_not_guessed():
    from transformers import BertConfig

    with pytest.raises(AdapterError, match="no adapter for model_type 'bert'"):
        get_adapter(BertConfig())


def test_adapter_rejects_unexpected_structure(small_spec):
    """A model whose real structure deviates from the GPT-2 layout must
    fail loudly rather than be partitioned with a module missing."""
    import torch
    from transformers import GPT2Config

    cfg = GPT2Config(n_layer=2, n_embd=64, n_head=4, vocab_size=100, n_positions=16)
    adapter = GPT2Adapter()
    model = adapter.build_meta_model(cfg)
    with torch.device("meta"):
        model.transformer.h[1].extra = torch.nn.Linear(64, 64)
    with pytest.raises(AdapterError, match="block 1 parameters differ"):
        adapter.describe("x", cfg, model)

    model = adapter.build_meta_model(cfg)
    with torch.device("meta"):
        model.transformer.adapter_layer = torch.nn.Linear(64, 64)
    with pytest.raises(AdapterError, match="not mapped to any component"):
        adapter.describe("x", cfg, model)


def test_adapter_detects_config_layer_mismatch():
    from transformers import GPT2Config

    cfg = GPT2Config(n_layer=3, n_embd=64, n_head=4, vocab_size=100, n_positions=16)
    adapter = GPT2Adapter()
    model = adapter.build_meta_model(cfg)
    del model.transformer.h[2]
    with pytest.raises(AdapterError, match="2 blocks but config.n_layer = 3"):
        adapter.describe("x", cfg, model)


def test_unavailable_model_reported_honestly():
    with pytest.raises(ModelInspectionError, match="could not load the configuration"):
        inspect_model("this-org/definitely-not-a-real-model-xyz", local_files_only=True)


def test_config_fingerprint_changes_with_config():
    from transformers import GPT2Config

    a = GPT2Adapter.fingerprint(GPT2Config(n_layer=2))
    assert a == GPT2Adapter.fingerprint(GPT2Config(n_layer=2))
    assert a != GPT2Adapter.fingerprint(GPT2Config(n_layer=3))
