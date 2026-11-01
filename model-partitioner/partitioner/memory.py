"""Memory estimation for a contiguous range of partition units.

Every figure is a MemoryItem tagged with its provenance (see
provenance.py); docs/MEMORY_ESTIMATION.md gives every formula.

What is CALCULATED (exact arithmetic from the real parameter shapes):
  - unquantized weights: numel x bytes(dtype)
  - KV cache tensors:    2 x batch x context x kv_heads x head_dim x bytes
What is ESTIMATED (formula with stated assumptions):
  - quantized weights (int8 per-channel / int4 grouped: packing + scales)
  - activation peaks (eager attention, no fused kernels assumed)
  - temporary load buffers (dtype conversion of the largest tensor)
What is MEASURED:
  - runtime overhead default: RSS of a Python process after importing
    torch + transformers and building GPT-2 (meta), on Computer 1.

Estimating int8/int4 memory does NOT mean quantized execution exists:
no quantized runtime is implemented in this project (Phase 6 will run
fp32/fp16/bf16 only unless that changes). Plans say so explicitly.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Dict, List, Optional, Tuple

from partitioner import model_spec as ms
from partitioner.provenance import MemoryItem, Provenance, weakest

DTYPE_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2}
QUANTIZATIONS = ("none", "int8", "int4")
MIB = 2**20

# Measured 2026-10-31 on Computer 1 (Python 3.13.3, torch 2.10.0+cu126,
# transformers 5.20.1): process RSS was 698-701 MiB after importing
# torch + transformers, loading the GPT-2 config and building the model
# on the meta device (the library cost a worker pays before any
# weight). Machine- and version-specific -- configurable.
RUNTIME_OVERHEAD_MEASURED_BYTES = 700 * MIB


class EstimationError(ValueError):
    pass


@dataclasses.dataclass(frozen=True)
class Precision:
    weight_dtype: str = "fp32"
    quantization: str = "none"  # "none" | "int8" | "int4"
    # Defaults: weight_dtype, or fp16 when weights are quantized.
    activation_dtype: Optional[str] = None
    kv_cache_dtype: Optional[str] = None
    quant_group_size: int = 128  # int4 group size
    quantize_embeddings: bool = False
    quantize_lm_head: bool = False

    def validate(self) -> None:
        problems = []
        if self.weight_dtype not in DTYPE_BYTES:
            problems.append(f"weight_dtype must be one of {sorted(DTYPE_BYTES)}, got {self.weight_dtype!r}")
        if self.quantization not in QUANTIZATIONS:
            problems.append(f"quantization must be one of {list(QUANTIZATIONS)}, got {self.quantization!r}")
        for field in ("activation_dtype", "kv_cache_dtype"):
            v = getattr(self, field)
            if v is not None and v not in DTYPE_BYTES:
                problems.append(f"{field} must be one of {sorted(DTYPE_BYTES)}, got {v!r}")
        if self.quant_group_size <= 0:
            problems.append("quant_group_size must be positive")
        if problems:
            raise EstimationError("; ".join(problems))

    @property
    def act(self) -> str:
        if self.activation_dtype:
            return self.activation_dtype
        return "fp16" if self.quantization != "none" else self.weight_dtype

    @property
    def kv(self) -> str:
        return self.kv_cache_dtype or self.act

    @property
    def unquantized_dtype(self) -> str:
        """dtype of parameters that are NOT quantized (norms, biases,
        embeddings) when quantization is on."""
        return self.act if self.quantization != "none" else self.weight_dtype

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d.update(
            activation_dtype=self.act,
            kv_cache_dtype=self.kv,
            quantization_runtime_implemented=False if self.quantization != "none" else None,
        )
        return d


@dataclasses.dataclass(frozen=True)
class Workload:
    batch_size: int = 1
    # Total tokens per sequence the KV cache must hold. Default: the
    # model's maximum (n_positions for GPT-2).
    context_length: Optional[int] = None
    # Tokens processed in one forward pass during prefill (memory peak).
    # Default: context_length -- the worst case.
    prompt_length: Optional[int] = None
    # What the time estimate measures: "decode" = one generated token
    # per sequence with a full KV cache; "prefill" = the prompt pass.
    cost_mode: str = "decode"
    # Logits kept in prefill: "all" positions (HF default) or "last".
    logits_positions: str = "all"

    def resolve(self, spec: ms.ModelSpec) -> "Workload":
        ctx = self.context_length or spec.max_positions
        prompt = self.prompt_length or ctx
        problems = []
        if self.batch_size <= 0:
            problems.append("batch_size must be positive")
        if ctx <= 0 or ctx > spec.max_positions:
            problems.append(f"context_length must be in 1..{spec.max_positions} for {spec.model_id}, got {ctx}")
        if prompt <= 0 or prompt > ctx:
            problems.append(f"prompt_length must be in 1..context_length ({ctx}), got {prompt}")
        if self.cost_mode not in ("decode", "prefill"):
            problems.append(f"cost_mode must be 'decode' or 'prefill', got {self.cost_mode!r}")
        if self.logits_positions not in ("all", "last"):
            problems.append(f"logits_positions must be 'all' or 'last', got {self.logits_positions!r}")
        if problems:
            raise EstimationError("; ".join(problems))
        return dataclasses.replace(self, context_length=ctx, prompt_length=prompt)


@dataclasses.dataclass(frozen=True)
class MemoryPolicy:
    runtime_overhead_bytes: int = RUNTIME_OVERHEAD_MEASURED_BYTES
    runtime_overhead_basis: str = (
        "MEASURED on Computer 1: RSS after importing torch 2.10 + transformers 5.20 and building GPT-2 "
        "on the meta device (698-701 MiB); assumed for every worker"
    )
    runtime_overhead_provenance: Provenance = Provenance.MEASURED
    # Fraction of each worker's usable memory the planner refuses to
    # use, ON TOP OF the Phase 3 margin already subtracted from
    # "usable" (see workers.py).
    extra_safety_fraction: float = 0.10
    include_kv_cache: bool = True
    include_load_temp: bool = True
    checkpoint_dtype: str = "fp32"  # how the weights are stored on disk

    def validate(self) -> None:
        if not 0 <= self.extra_safety_fraction < 1:
            raise EstimationError("extra_safety_fraction must be in [0, 1)")
        if self.runtime_overhead_bytes < 0:
            raise EstimationError("runtime_overhead_bytes must not be negative")
        if self.checkpoint_dtype not in DTYPE_BYTES:
            raise EstimationError(f"checkpoint_dtype must be one of {sorted(DTYPE_BYTES)}")

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["runtime_overhead_provenance"] = self.runtime_overhead_provenance.value
        return d


# ---------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------


def param_bytes(p: ms.ParamInfo, component_kind: str, precision: Precision) -> Tuple[int, Provenance, str]:
    """Bytes to hold one parameter at the chosen precision."""
    quantize = (
        precision.quantization != "none"
        and p.kind == ms.LINEAR_WEIGHT
        and (component_kind == ms.TRANSFORMER_BLOCK or (component_kind == ms.LM_HEAD and precision.quantize_lm_head))
    ) or (precision.quantization != "none" and p.kind == ms.EMBEDDING_TABLE and precision.quantize_embeddings)
    if not quantize:
        dtype = precision.unquantized_dtype
        return p.numel * DTYPE_BYTES[dtype], Provenance.CALCULATED, f"{p.numel} x {DTYPE_BYTES[dtype]} B ({dtype})"
    # Per-output-channel (int8) / per-group (int4) fp16 scales. The
    # channel axis is the last for GPT-2's Conv1D (in, out) layout.
    out_channels = p.shape[-1] if len(p.shape) >= 2 else 1
    if precision.quantization == "int8":
        n = p.numel + out_channels * 2
        return n, Provenance.ESTIMATED, f"int8: {p.numel} x 1 B + {out_channels} fp16 per-channel scales"
    groups = math.ceil(p.numel / precision.quant_group_size)
    n = math.ceil(p.numel / 2) + groups * 4
    return n, Provenance.ESTIMATED, (
        f"int4: {p.numel} x 0.5 B + {groups} groups of {precision.quant_group_size} x (fp16 scale + fp16 zero)"
    )


@dataclasses.dataclass(frozen=True)
class UnitMemory:
    unit: ms.Unit
    owned_weight_bytes: int  # parameters this unit stores itself
    weight_provenance: Provenance
    # tied parameter -> (owner param, bytes it needs if its owner isn't
    # on the same stage)
    tied: Tuple[Tuple[str, str, int], ...]
    kv_bytes: int
    activation_bytes: int  # peak while this unit runs (prefill)
    largest_param_numel: int


def unit_memory(spec: ms.ModelSpec, unit: ms.Unit, precision: Precision, workload: Workload) -> UnitMemory:
    a = DTYPE_BYTES[precision.act]
    B, T, L = workload.batch_size, workload.prompt_length, workload.context_length
    h, inner = spec.hidden_size, spec.intermediate_size
    att = spec.attention

    owned, provs, tied, largest = 0, [], [], 0
    for comp in spec.unit_components(unit):
        for p in comp.params:
            nbytes, prov, _ = param_bytes(p, comp.kind, precision)
            largest = max(largest, p.numel)
            if p.tied_to is not None:
                tied.append((p.name, p.tied_to, nbytes))
            else:
                owned += nbytes
                provs.append(prov)

    kv = 0
    act = 0
    if unit.kind == ms.TRANSFORMER_BLOCK:
        kv = 2 * B * L * att.n_kv_heads * att.head_dim * DTYPE_BYTES[precision.kv]
        hidden = B * T * h
        attention_phase = 2 * hidden + 3 * hidden + hidden + 2 * B * att.n_heads * T * L
        mlp_phase = 2 * hidden + 2 * B * T * inner
        act = max(attention_phase, mlp_phase) * a
    elif unit.kind == ms.EMBEDDING:
        act = B * T * h * a
    elif unit.kind == ms.FINAL_NORM:
        act = 2 * B * T * h * a
    elif unit.kind == ms.LM_HEAD:
        positions = T if workload.logits_positions == "all" else 1
        act = (B * T * h + B * positions * spec.vocab_size) * a
    return UnitMemory(unit, owned, weakest(provs) if provs else Provenance.CALCULATED, tuple(tied), kv, act, largest)


@dataclasses.dataclass(frozen=True)
class StageMemory:
    """Requirement for running units [start, end) on one worker.

    device_bytes go to the compute device's memory pool (RAM for a CPU
    stage, VRAM for a GPU stage); host_bytes always go to RAM."""

    start: int
    end: int
    items: Tuple[MemoryItem, ...]
    device_bytes: int
    host_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.device_bytes + self.host_bytes

    @property
    def provenance(self) -> Provenance:
        return weakest(i.provenance for i in self.items)

    def to_dict(self) -> dict:
        return {
            "device_bytes": self.device_bytes,
            "host_bytes": self.host_bytes,
            "total_bytes": self.total_bytes,
            "provenance": self.provenance.value,
            "items": [i.to_dict() for i in self.items],
        }


class MemoryEstimator:
    """Precomputes per-unit figures once; stage figures are then cheap,
    which the optimizer relies on (it evaluates every range)."""

    def __init__(self, spec: ms.ModelSpec, precision: Precision, workload: Workload, policy: MemoryPolicy):
        precision.validate()
        policy.validate()
        self.spec, self.precision, self.policy = spec, precision, policy
        self.workload = workload.resolve(spec)
        self.units = [unit_memory(spec, u, precision, self.workload) for u in spec.units]
        self._owner_unit: Dict[str, int] = {}
        for um in self.units:
            for comp in spec.unit_components(um.unit):
                for p in comp.params:
                    if p.tied_to is None:
                        self._owner_unit[p.name] = um.unit.index

    def stage(self, start: int, end: int) -> StageMemory:
        if not 0 <= start < end <= len(self.units):
            raise EstimationError(f"invalid unit range [{start}, {end})")
        rng = self.units[start:end]
        a = DTYPE_BYTES[self.precision.act]
        wl = self.workload
        items: List[MemoryItem] = []

        weights = sum(u.owned_weight_bytes for u in rng)
        wprov = weakest(u.weight_provenance for u in rng)
        q = self.precision.quantization
        items.append(MemoryItem(
            "weights", weights, wprov,
            f"parameters of units {start}..{end - 1} at {self.precision.unquantized_dtype if q == 'none' else q + ' (linear weights) / ' + self.precision.unquantized_dtype}",
        ))

        dup = 0
        dup_names = []
        for u in rng:
            for tied_name, owner, nbytes in u.tied:
                if not start <= self._owner_unit[owner] < end:
                    dup += nbytes
                    dup_names.append(f"{tied_name} (tied to {owner})")
        if dup:
            items.append(MemoryItem(
                "tied_weight_copy", dup, Provenance.CALCULATED,
                f"{', '.join(dup_names)}: owner is on another stage, so this stage holds its own copy",
            ))

        kv = sum(u.kv_bytes for u in rng) if self.policy.include_kv_cache else 0
        if kv:
            items.append(MemoryItem(
                "kv_cache", kv, Provenance.CALCULATED,
                f"2 x batch {wl.batch_size} x context {wl.context_length} x {self.spec.attention.n_kv_heads} kv heads "
                f"x head_dim {self.spec.attention.head_dim} x {DTYPE_BYTES[self.precision.kv]} B per block",
            ))

        act = max(u.activation_bytes for u in rng)
        items.append(MemoryItem(
            "activations_peak", act, Provenance.ESTIMATED,
            f"largest single-unit peak for a {wl.prompt_length}-token prefill (eager attention, freed between layers)",
        ))
        if end < len(self.units):
            send = wl.batch_size * wl.prompt_length * self.spec.hidden_size * a
            items.append(MemoryItem(
                "outgoing_hidden_state_buffer", send, Provenance.ESTIMATED,
                f"batch x {wl.prompt_length} tokens x hidden {self.spec.hidden_size} x {a} B held for the next stage",
            ))
        else:
            send = 0
        device = weights + dup + kv + act + send

        host = 0
        if self.policy.include_load_temp:
            largest = max(u.largest_param_numel for u in rng)
            temp = largest * DTYPE_BYTES[self.policy.checkpoint_dtype] if largest else 0
            if temp:
                items.append(MemoryItem(
                    "load_temp_buffer", temp, Provenance.ESTIMATED,
                    f"largest tensor ({largest} values) in checkpoint dtype {self.policy.checkpoint_dtype} during load/conversion",
                ))
                host += temp
        items.append(MemoryItem(
            "runtime_overhead", self.policy.runtime_overhead_bytes, self.policy.runtime_overhead_provenance,
            self.policy.runtime_overhead_basis,
        ))
        host += self.policy.runtime_overhead_bytes
        return StageMemory(start, end, tuple(items), device, host)

    def whole_model_weights(self) -> Dict[str, int]:
        """Unique parameter bytes, ignoring stage placement."""
        return {"bytes": sum(u.owned_weight_bytes for u in self.units)}

    def unit_table(self) -> List[dict]:
        rows = []
        for um in self.units:
            rows.append({
                "unit": um.unit.index,
                "name": um.unit.name,
                "kind": um.unit.kind,
                "params": sum(c.owned_numel for c in self.spec.unit_components(um.unit)),
                "tied_params": sum(c.tied_numel for c in self.spec.unit_components(um.unit)),
                "weight_bytes": um.owned_weight_bytes,
                "weight_provenance": um.weight_provenance.value,
                "kv_cache_bytes": um.kv_bytes,
                "activation_peak_bytes": um.activation_bytes,
            })
        return rows
