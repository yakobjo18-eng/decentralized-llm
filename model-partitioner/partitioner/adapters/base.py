"""Architecture adapter interface.

An adapter turns one family of Hugging Face models into a ModelSpec. It
is the ONLY place that knows module names, how layers are numbered, and
which weights are tied. Adding a new sequential decoder architecture
(e.g. Llama: embed_tokens / layers[i] / norm / lm_head, grouped-query
attention) means writing one new adapter -- the memory model, cost
model, optimizer, plan and validator are architecture-neutral.

Adapters must be strict: every module of the real model must map to
exactly one component, and shapes must match what the config implies.
Anything unexpected raises AdapterError rather than producing a
silently incomplete partition.
"""

from __future__ import annotations

import abc
import hashlib
import json
from typing import Any, Dict


class AdapterError(ValueError):
    """The model doesn't have the structure this adapter expects, or its
    configuration is invalid."""


class ArchitectureAdapter(abc.ABC):
    name: str = ""
    version: int = 1
    model_types: tuple = ()

    def supports(self, config: Any) -> bool:
        return getattr(config, "model_type", None) in self.model_types

    @abc.abstractmethod
    def validate_config(self, config: Any) -> None:
        """Raise AdapterError for configurations this adapter can't
        describe correctly (before anything is instantiated)."""

    @abc.abstractmethod
    def build_meta_model(self, config: Any):
        """Instantiate the model on the meta device: real module
        structure and parameter shapes, zero bytes of weights."""

    @abc.abstractmethod
    def describe(self, model_id: str, config: Any, model: Any):
        """Return a ModelSpec for the instantiated meta model."""

    @staticmethod
    def fingerprint(config: Any) -> str:
        """sha256 over the full config (canonical JSON), so a plan can
        be checked against the exact configuration it was made for."""
        data: Dict[str, Any] = config.to_dict() if hasattr(config, "to_dict") else dict(config)
        # Fields that describe the serialization, not the model.
        for volatile in ("transformers_version", "_name_or_path", "_commit_hash", "_attn_implementation_autoset"):
            data.pop(volatile, None)
        canonical = json.dumps(data, sort_keys=True, default=str, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
