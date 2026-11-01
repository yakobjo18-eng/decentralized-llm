"""Plan dtype names <-> torch dtypes. Phase 5 executes fp32 only; the
names match partitioner.memory.DTYPE_BYTES."""

from __future__ import annotations

EXECUTABLE_DTYPES = ("fp32",)


def torch_dtype(name: str):
    import torch

    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16, "int64": torch.int64}[name]


def torch_dtype_name(dtype) -> str:
    import torch

    names = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16", torch.int64: "int64"}
    return names.get(dtype, str(dtype).replace("torch.", ""))
