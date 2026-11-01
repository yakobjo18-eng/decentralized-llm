"""Built-in executable test model: builtin:tiny-gpt2-v1.

A real GPT-2 architecture at toy size, so that a Phase 4 plan for it is
produced by the unchanged GPT2Adapter (same unit chain: embeddings,
block.0..block.3, final_norm, lm_head) and can be executed for real:

    n_layer 4, n_embd 32, n_head 4 (head_dim 8), n_inner 128,
    vocab_size 128, n_positions 32, gelu_new, tied lm_head <-> wte
    56,000 parameters = 224,000 bytes in fp32

Weights are NOT random and NOT downloaded: every parameter is a fixed
formula of its name and element index (WEIGHTS_VERSION below), so any
machine rebuilds bit-identical fp32 weights for exactly the components
it needs. Values are scaled to stay in GPT-2's numeric range; they are
not trained and the logits carry no linguistic meaning.

The math is written out explicitly (no Hugging Face forward pass):
Conv1D layout (x @ W + b), pre-LayerNorm blocks, causal eager
attention scaled by 1/sqrt(head_dim), tanh-approximated GELU, no
dropout (inference). Tests cross-check it against an independent NumPy
implementation and against transformers' GPT2LMHeadModel loaded with
the same weights.
"""

from __future__ import annotations

import functools
import math
import zlib
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

from partitioner import model_spec as ms
from partitioner.execution.dtypes import EXECUTABLE_DTYPES, torch_dtype, torch_dtype_name
from partitioner.execution.interface import (
    HIDDEN_STATES,
    INPUT_STAGE,
    LOGITS,
    STAGE_INTERFACE_VERSION,
    TOKEN_IDS,
    Activation,
    ActivationMeta,
    InterfaceError,
    StageExecutor,
    StageSpec,
    expected_output,
)

MODEL_ID = "builtin:tiny-gpt2-v1"
WEIGHTS_VERSION = "formula-sin-v1"
DESCRIPTION = "4-layer GPT-2 architecture, hidden 32, vocab 128, 32 positions; formula weights (untrained)"


def config():
    from transformers import GPT2Config

    return GPT2Config(
        n_layer=4, n_embd=32, n_head=4, vocab_size=128, n_positions=32,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
        # GPT-2's defaults (50256) lie outside this vocabulary.
        bos_token_id=127, eos_token_id=127,
    )


@functools.lru_cache(maxsize=1)
def spec() -> ms.ModelSpec:
    """The Phase 4 ModelSpec, produced by the real GPT2Adapter from the
    config above (meta device; nothing downloaded)."""
    from partitioner.inspection import inspect_model

    return inspect_model(MODEL_ID, config=config(), check_checkpoint=False)


# ---------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------


def deterministic_param(name: str, shape: Tuple[int, ...], kind: str) -> torch.Tensor:
    """formula-sin-v1: a smooth wave whose phase is derived from the
    parameter's name (crc32), computed in float64 and rounded to fp32."""
    n = math.prod(shape)
    phase = (zlib.crc32(name.encode("utf-8")) % 100003) / 100003 * 2 * math.pi
    i = torch.arange(n, dtype=torch.float64)
    wave = torch.sin(0.37 * i + phase) * torch.cos(0.011 * i + 0.5 * phase)
    if kind == ms.NORM and name.endswith(".weight"):
        values = 1.0 + 0.1 * wave
    elif kind in (ms.NORM, ms.BIAS):
        values = 0.02 * wave
    elif kind == ms.EMBEDDING_TABLE:
        values = 0.5 * wave
    else:  # linear weights
        values = 0.15 * wave
    return values.reshape(shape).to(torch.float32)


def stage_weights(model_spec: ms.ModelSpec, start: int, end: int) -> Dict[str, torch.Tensor]:
    """Exactly the tensors units [start, end) need, keyed by parameter
    name. A tied parameter whose owner is on this stage aliases the
    owner's tensor; otherwise the stage gets its own copy (the plan's
    `tied_weight_copy` memory item)."""
    units = model_spec.units[start:end]
    comps = [c for u in units for c in model_spec.unit_components(u)]
    all_params = {p.name: (p, c.kind) for c in model_spec.components for p in c.params}
    out: Dict[str, torch.Tensor] = {}
    for c in comps:
        for p in c.params:
            if p.tied_to is None:
                out[p.name] = deterministic_param(p.name, p.shape, p.kind)
    for c in comps:
        for p in c.params:
            if p.tied_to is not None:
                if p.tied_to in out:
                    out[p.name] = out[p.tied_to]
                else:
                    owner, _ = all_params[p.tied_to]
                    out[p.name] = deterministic_param(owner.name, owner.shape, owner.kind)
    return out


def full_weights(model_spec: Optional[ms.ModelSpec] = None) -> Dict[str, torch.Tensor]:
    model_spec = model_spec or spec()
    return stage_weights(model_spec, 0, len(model_spec.units))


def weight_bytes(model_spec: ms.ModelSpec, start: int, end: int) -> int:
    """fp32 bytes stage_weights() would allocate -- computed from the
    spec's shapes, before allocating anything."""
    names = set()
    total = 0
    for u in model_spec.units[start:end]:
        for c in model_spec.unit_components(u):
            for p in c.params:
                names.add(p.name)
                total += p.numel * 4
    # An aliased tied parameter costs nothing extra.
    for tied, owner in model_spec.tied_weights:
        if tied in names and owner in names:
            total -= math.prod(next(p.shape for c in model_spec.components for p in c.params if p.name == tied)) * 4
    return total


# ---------------------------------------------------------------------
# Unit math
# ---------------------------------------------------------------------


def _block(x: torch.Tensor, w: Dict[str, torch.Tensor], p: str, n_head: int, eps: float) -> torch.Tensor:
    B, T, H = x.shape
    hd = H // n_head
    h = F.layer_norm(x, (H,), w[p + "ln_1.weight"], w[p + "ln_1.bias"], eps)
    qkv = h @ w[p + "attn.c_attn.weight"] + w[p + "attn.c_attn.bias"]
    q, k, v = (t.reshape(B, T, n_head, hd).transpose(1, 2) for t in qkv.split(H, dim=2))
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
    causal = torch.ones(T, T, dtype=torch.bool).tril()
    scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
    y = (scores.softmax(dim=-1) @ v).transpose(1, 2).reshape(B, T, H)
    x = x + (y @ w[p + "attn.c_proj.weight"] + w[p + "attn.c_proj.bias"])
    h = F.layer_norm(x, (H,), w[p + "ln_2.weight"], w[p + "ln_2.bias"], eps)
    m = F.gelu(h @ w[p + "mlp.c_fc.weight"] + w[p + "mlp.c_fc.bias"], approximate="tanh")
    return x + (m @ w[p + "mlp.c_proj.weight"] + w[p + "mlp.c_proj.bias"])


def run_unit(model_spec: ms.ModelSpec, unit: ms.Unit, w: Dict[str, torch.Tensor], x: torch.Tensor,
             positions_start: int = 0) -> torch.Tensor:
    eps = model_spec.config["layer_norm_epsilon"]
    if unit.kind == ms.EMBEDDING:
        T = x.shape[1]
        return F.embedding(x, w["transformer.wte.weight"]) + w["transformer.wpe.weight"][positions_start:positions_start + T]
    if unit.kind == ms.TRANSFORMER_BLOCK:
        (component,) = unit.components
        return _block(x, w, component + ".", model_spec.attention.n_heads, eps)
    if unit.kind == ms.FINAL_NORM:
        return F.layer_norm(x, (model_spec.hidden_size,), w["transformer.ln_f.weight"], w["transformer.ln_f.bias"], eps)
    if unit.kind == ms.LM_HEAD:
        return x @ w["lm_head.weight"].T
    raise InterfaceError(f"unit {unit.name}: kind {unit.kind!r} has no implementation in {MODEL_ID}")


def reference_forward(token_ids: torch.Tensor, weights: Optional[Dict[str, torch.Tensor]] = None,
                      model_spec: Optional[ms.ModelSpec] = None) -> torch.Tensor:
    """The UNPARTITIONED reference: every unit in order, one weight dict,
    no stages, no transport, no plan. Returns logits [batch, tokens, vocab]."""
    model_spec = model_spec or spec()
    weights = weights if weights is not None else full_weights(model_spec)
    x = token_ids
    with torch.inference_mode():
        for unit in model_spec.units:
            x = run_unit(model_spec, unit, weights, x)
    return x


# ---------------------------------------------------------------------
# Stage executor
# ---------------------------------------------------------------------


def output_kind(model_spec: ms.ModelSpec, end: int) -> str:
    return LOGITS if model_spec.units[end - 1].kind == ms.LM_HEAD else HIDDEN_STATES


def check_stage_spec(model_spec: ms.ModelSpec, s: StageSpec) -> List[str]:
    """Every way a StageSpec can disagree with this model."""
    problems = []
    if s.interface_version != STAGE_INTERFACE_VERSION:
        problems.append(f"stage interface_version {s.interface_version} != supported {STAGE_INTERFACE_VERSION}")
    if s.model_id != model_spec.model_id:
        problems.append(f"stage is for model {s.model_id!r}, executor is {model_spec.model_id!r}")
    if s.config_fingerprint != model_spec.config_fingerprint:
        problems.append("stage config_fingerprint does not match the model's configuration")
    U = len(model_spec.units)
    start, end = s.unit_range
    if not (isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= U):
        problems.append(f"unit_range {list(s.unit_range)} is not a non-empty range within [0, {U})")
        return problems
    units = model_spec.units[start:end]
    if tuple(s.unit_names) != tuple(u.name for u in units):
        problems.append(f"unit_names {list(s.unit_names)} don't match unit_range {[start, end]}")
    if tuple(s.components) != tuple(c for u in units for c in u.components):
        problems.append(f"components don't match unit_range {[start, end]}")
    want_in = TOKEN_IDS if start == 0 else HIDDEN_STATES
    if s.input_kind != want_in:
        problems.append(f"input_kind {s.input_kind!r} != {want_in!r} for a stage starting at unit {start}")
    if s.output_kind != output_kind(model_spec, end):
        problems.append(f"output_kind {s.output_kind!r} != {output_kind(model_spec, end)!r} for a stage ending at unit {end}")
    if s.activation_dtype not in EXECUTABLE_DTYPES:
        problems.append(f"activation dtype {s.activation_dtype!r} is not executable (supported: {list(EXECUTABLE_DTYPES)})")
    if model_spec.config.get("activation_function") != "gelu_new":
        problems.append(f"activation_function {model_spec.config.get('activation_function')!r} is not implemented")
    return problems


class TinyGPT2Stage(StageExecutor):
    def __init__(self, stage_spec: StageSpec, model_spec: Optional[ms.ModelSpec] = None):
        self.model_spec = model_spec or spec()
        problems = check_stage_spec(self.model_spec, stage_spec)
        if problems:
            raise InterfaceError(f"stage {stage_spec.stage_index}: " + "; ".join(problems))
        self.spec = stage_spec
        start, end = stage_spec.unit_range
        self.units = self.model_spec.units[start:end]
        self.weights = stage_weights(self.model_spec, start, end)

    def forward(self, activation: Activation) -> Activation:
        activation.check()
        s, m = self.spec, activation.meta
        problems = []
        if m.kind != s.input_kind:
            problems.append(f"expected a {s.input_kind} activation, got {m.kind}")
        want_from = INPUT_STAGE if s.stage_index == 0 else s.stage_index - 1
        if m.produced_by_stage != want_from:
            problems.append(f"activation comes from stage {m.produced_by_stage}, expected {want_from}")
        if m.positions_start != 0:
            problems.append("positions_start must be 0: only a single prefill pass is implemented (no KV cache)")
        if len(m.shape) < 2:
            problems.append(f"activation shape {list(m.shape)} has no [batch, tokens] dimensions")
        else:
            B, T = m.shape[0], m.shape[1]
            want_shape = expected_output(s.input_kind, B, T, self.model_spec.hidden_size, self.model_spec.vocab_size)
            want_dtype = "int64" if s.input_kind == TOKEN_IDS else s.activation_dtype
            if tuple(m.shape) != want_shape:
                problems.append(f"shape {list(m.shape)} != expected {list(want_shape)}")
            if m.dtype != want_dtype:
                problems.append(f"dtype {m.dtype} != expected {want_dtype}")
            if T > self.model_spec.max_positions:
                problems.append(f"{T} tokens > the model's {self.model_spec.max_positions} positions")
            if s.input_kind == TOKEN_IDS and activation.tensor.numel():
                lo, hi = int(activation.tensor.min()), int(activation.tensor.max())
                if lo < 0 or hi >= self.model_spec.vocab_size:
                    problems.append(f"token ids must be in [0, {self.model_spec.vocab_size}), got [{lo}, {hi}]")
        if problems:
            raise InterfaceError(f"stage {s.stage_index} rejected its input: " + "; ".join(problems))

        x = activation.tensor
        if s.input_kind != TOKEN_IDS:
            x = x.to(torch_dtype(s.activation_dtype))
        with torch.inference_mode():
            for unit in self.units:
                x = run_unit(self.model_spec, unit, self.weights, x, m.positions_start)
        meta = ActivationMeta(STAGE_INTERFACE_VERSION, m.request_id, s.output_kind, tuple(x.shape),
                              torch_dtype_name(x.dtype), s.stage_index, m.positions_start)
        return Activation(meta, x)


def make_stage_spec(model_spec: ms.ModelSpec, stage_index: int, n_stages: int, start: int, end: int,
                    activation_dtype: str = "fp32") -> StageSpec:
    units = model_spec.units[start:end]
    return StageSpec(
        interface_version=STAGE_INTERFACE_VERSION,
        model_id=model_spec.model_id,
        config_fingerprint=model_spec.config_fingerprint,
        stage_index=stage_index,
        n_stages=n_stages,
        unit_range=(start, end),
        unit_names=tuple(u.name for u in units),
        components=tuple(c for u in units for c in u.components),
        input_kind=TOKEN_IDS if start == 0 else HIDDEN_STATES,
        output_kind=output_kind(model_spec, end) if 0 < end <= len(model_spec.units) else HIDDEN_STATES,
        activation_dtype=activation_dtype,
    )


def token_tensor(token_ids: Iterable[Iterable[int]]) -> torch.Tensor:
    return torch.tensor([list(row) for row in token_ids], dtype=torch.int64)
