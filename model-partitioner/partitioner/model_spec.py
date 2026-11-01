"""Architecture-neutral description of an inspected model.

The partitioning algorithm only ever sees a ModelSpec: an ordered chain
of *units* (the indivisible pieces a stage boundary can fall between),
each made of *components* (real module paths in the model). Nothing
downstream knows GPT-2's layer count or module names -- an adapter
(partitioner/adapters/) produces the spec from the real model.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple

# Component kinds the memory and cost models understand.
EMBEDDING = "embedding"
POSITIONAL_EMBEDDING = "positional_embedding"
PARAMETERLESS = "parameterless"  # e.g. dropout: assigned, holds no weights
TRANSFORMER_BLOCK = "transformer_block"
FINAL_NORM = "final_norm"
LM_HEAD = "lm_head"

# Parameter kinds (drive quantization: only linear weights are quantized).
LINEAR_WEIGHT = "linear_weight"
BIAS = "bias"
NORM = "norm"
EMBEDDING_TABLE = "embedding_table"


@dataclasses.dataclass(frozen=True)
class ParamInfo:
    name: str  # full parameter path, e.g. "transformer.h.0.attn.c_attn.weight"
    shape: Tuple[int, ...]
    numel: int
    kind: str
    # Set when this parameter is the SAME tensor as another one (weight
    # tying); its storage belongs to `tied_to`.
    tied_to: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class Component:
    name: str  # module path, e.g. "transformer.h.0"
    kind: str
    params: Tuple[ParamInfo, ...]
    block_index: Optional[int] = None  # for transformer blocks

    @property
    def owned_numel(self) -> int:
        return sum(p.numel for p in self.params if p.tied_to is None)

    @property
    def tied_numel(self) -> int:
        return sum(p.numel for p in self.params if p.tied_to is not None)


@dataclasses.dataclass(frozen=True)
class Unit:
    """Smallest piece a stage boundary can split around."""

    index: int
    name: str
    kind: str
    components: Tuple[str, ...]
    block_index: Optional[int] = None


@dataclasses.dataclass(frozen=True)
class AttentionShape:
    n_heads: int
    n_kv_heads: int  # == n_heads for GPT-2; fewer under grouped-query attention
    head_dim: int


@dataclasses.dataclass(frozen=True)
class CheckpointInfo:
    """What the real published checkpoint contains, read from its
    safetensors header only (no weights downloaded)."""

    status: str  # "verified" | "mismatch" | "unavailable"
    detail: str
    files: Tuple[str, ...] = ()
    stored_dtypes: Tuple[str, ...] = ()
    stored_values: int = 0
    stored_bytes: int = 0
    extra_tensors: Tuple[str, ...] = ()  # stored but not model parameters
    extra_values: int = 0
    missing_params: Tuple[str, ...] = ()
    shape_mismatches: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["extra_tensors_sample"] = list(self.extra_tensors[:5])
        d["extra_tensor_count"] = len(self.extra_tensors)
        del d["extra_tensors"]
        return d


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    model_id: str
    architecture: str  # config.model_type, e.g. "gpt2"
    adapter: str
    adapter_version: int
    config: Dict[str, object]  # the architecture-relevant config fields
    config_fingerprint: str  # sha256 of the full config, canonical JSON
    n_layers: int
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    max_positions: int
    attention: AttentionShape
    total_params: int  # unique parameters (tied counted once)
    components: Tuple[Component, ...]
    units: Tuple[Unit, ...]
    tied_weights: Tuple[Tuple[str, str], ...]  # (tied param, owner param)
    checkpoint: Optional[CheckpointInfo] = None
    inspection: Dict[str, object] = dataclasses.field(default_factory=dict)

    def component(self, name: str) -> Component:
        return self._components_by_name()[name]

    def _components_by_name(self) -> Dict[str, Component]:
        return {c.name: c for c in self.components}

    def unit_components(self, unit: Unit) -> List[Component]:
        by_name = self._components_by_name()
        return [by_name[n] for n in unit.components]

    def param_owner(self) -> Dict[str, str]:
        """param name -> component name that stores it."""
        return {p.name: c.name for c in self.components for p in c.params if p.tied_to is None}

    def summary(self) -> dict:
        return {
            "model_id": self.model_id,
            "architecture": self.architecture,
            "adapter": f"{self.adapter} v{self.adapter_version}",
            "n_layers": self.n_layers,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "n_heads": self.attention.n_heads,
            "n_kv_heads": self.attention.n_kv_heads,
            "head_dim": self.attention.head_dim,
            "vocab_size": self.vocab_size,
            "max_positions": self.max_positions,
            "total_params": self.total_params,
            "units": len(self.units),
            "tied_weights": [list(t) for t in self.tied_weights],
            "config_fingerprint": self.config_fingerprint,
        }
