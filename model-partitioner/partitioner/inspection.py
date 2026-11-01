"""Model inspection: configuration first, weights never.

  1. Load the model's config (config.json -- a few hundred bytes, from
     the local Hugging Face cache or the Hub).
  2. Instantiate the real model class on PyTorch's meta device: the
     actual module tree and exact parameter shapes, no weight memory.
  3. Optionally read the published checkpoint's safetensors HEADER
     (a few KB, via an HTTP range request) and cross-check it against
     the model: tensor names, shapes, stored dtype.

Measured on Computer 1 (2026-10-31): steps 1-2 cost ~2 MiB on top of
the ~700 MiB that importing torch + transformers itself costs; no
weight file is downloaded. If the config isn't cached and the Hub is
unreachable, inspection fails with that reason -- results are never
fabricated.
"""

from __future__ import annotations

import gc
import math
import os
from typing import Any, Optional

from partitioner import model_spec as ms
from partitioner.adapters import get_adapter


class ModelInspectionError(RuntimeError):
    pass


def _rss_mib() -> Optional[float]:
    try:
        import psutil

        return round(psutil.Process(os.getpid()).memory_info().rss / 2**20, 1)
    except Exception:
        return None


def load_config(model_id: str, local_files_only: bool = False, revision: Optional[str] = None):
    try:
        from transformers import AutoConfig

        return AutoConfig.from_pretrained(model_id, local_files_only=local_files_only, revision=revision)
    except Exception as exc:
        where = "the local Hugging Face cache" if local_files_only else "the local cache or the Hugging Face Hub"
        raise ModelInspectionError(f"could not load the configuration of {model_id!r} from {where}: {exc}") from exc


def read_checkpoint_info(model_id: str, spec: ms.ModelSpec, revision: Optional[str] = None) -> ms.CheckpointInfo:
    """Cross-checks the real checkpoint's safetensors header against the
    inspected model. Never raises: an unreachable Hub yields
    status="unavailable" with the reason."""
    try:
        from huggingface_hub import get_safetensors_metadata

        meta = get_safetensors_metadata(model_id, revision=revision)
    except Exception as exc:
        return ms.CheckpointInfo(status="unavailable", detail=f"checkpoint header not readable: {type(exc).__name__}: {exc}"[:400])

    stored = {}
    for _fname, fmeta in meta.files_metadata.items():
        for tname, tinfo in fmeta.tensors.items():
            stored[tname] = tinfo
    dtype_bytes = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}

    # Checkpoints often omit the backbone prefix ("h.0..." for
    # "transformer.h.0...") and omit tied copies.
    def stored_key(param_name: str) -> Optional[str]:
        if param_name in stored:
            return param_name
        short = param_name.split(".", 1)[1] if "." in param_name else param_name
        return short if short in stored else None

    matched = set()
    missing, mismatched = [], []
    for comp in spec.components:
        for p in comp.params:
            if p.tied_to is not None:
                continue  # stored once, under its owner
            key = stored_key(p.name)
            if key is None:
                missing.append(p.name)
                continue
            matched.add(key)
            if tuple(stored[key].shape) != p.shape:
                mismatched.append(f"{p.name}: checkpoint {tuple(stored[key].shape)} vs model {p.shape}")
    extra = sorted(set(stored) - matched)
    values = sum(math.prod(t.shape) for t in stored.values())
    nbytes = sum(math.prod(t.shape) * dtype_bytes.get(t.dtype, 0) for t in stored.values())
    status = "verified" if not missing and not mismatched else "mismatch"
    detail = (
        "every model parameter is present in the checkpoint with an identical shape"
        if status == "verified"
        else f"{len(missing)} parameters missing, {len(mismatched)} shape mismatches"
    )
    if extra:
        detail += f"; checkpoint also stores {len(extra)} tensors that are not model parameters (e.g. {extra[0]})"
    return ms.CheckpointInfo(
        status=status,
        detail=detail,
        files=tuple(sorted(meta.files_metadata)),
        stored_dtypes=tuple(sorted({t.dtype for t in stored.values()})),
        stored_values=values,
        stored_bytes=nbytes,
        extra_tensors=tuple(extra),
        extra_values=sum(math.prod(stored[k].shape) for k in extra),
        missing_params=tuple(missing),
        shape_mismatches=tuple(mismatched),
    )


def inspect_model(
    model_id: Optional[str] = None,
    config: Any = None,
    check_checkpoint: bool = True,
    local_files_only: bool = False,
    revision: Optional[str] = None,
) -> ms.ModelSpec:
    """Inspects a model by Hub id (config downloaded/cached) or from a
    config object (offline; used by tests with synthetic configs)."""
    if config is None:
        if not model_id:
            raise ModelInspectionError("either model_id or config is required")
        config = load_config(model_id, local_files_only=local_files_only, revision=revision)
    adapter = get_adapter(config)
    adapter.validate_config(config)

    rss_before = _rss_mib()
    model = adapter.build_meta_model(config)
    try:
        spec = adapter.describe(model_id or f"<config:{config.model_type}>", config, model)
    finally:
        del model
        gc.collect()
    rss_after = _rss_mib()

    import torch
    import transformers

    inspection = {
        "method": "meta-device instantiation of the real model class from its configuration; no weights loaded",
        "rss_before_mib": rss_before,
        "rss_after_mib": rss_after,
        "rss_provenance": "MEASURED (this process)",
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
    }
    checkpoint = None
    if check_checkpoint and model_id:
        checkpoint = read_checkpoint_info(model_id, spec, revision=revision)
    import dataclasses

    return dataclasses.replace(spec, checkpoint=checkpoint, inspection=inspection)
