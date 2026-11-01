"""Registry of architecture adapters. GPT-2 is the only supported
architecture; any other model_type is rejected explicitly rather than
partitioned by guesswork."""

from __future__ import annotations

from typing import Any, List

from partitioner.adapters.base import AdapterError, ArchitectureAdapter
from partitioner.adapters.gpt2 import GPT2Adapter

ADAPTERS: List[ArchitectureAdapter] = [GPT2Adapter()]


def get_adapter(config: Any) -> ArchitectureAdapter:
    for adapter in ADAPTERS:
        if adapter.supports(config):
            return adapter
    supported = sorted(t for a in ADAPTERS for t in a.model_types)
    raise AdapterError(
        f"no adapter for model_type {getattr(config, 'model_type', None)!r}; supported: {supported}. "
        "Each architecture needs its own adapter (see partitioner/adapters/base.py)."
    )


__all__ = ["ADAPTERS", "AdapterError", "ArchitectureAdapter", "GPT2Adapter", "get_adapter"]
