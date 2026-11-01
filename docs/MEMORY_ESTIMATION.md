# Memory Estimation (Phase 4)

How `model-partitioner` estimates the memory a contiguous range of
model units needs on one worker. Code: `model-partitioner/partitioner/memory.py`.

## Provenance: every number says where it came from

| Label | Meaning | Examples |
|---|---|---|
| **MEASURED** | observed on a real machine | runtime overhead default (Computer 1) |
| **CALCULATED** | exact arithmetic from known facts | unquantized weights (real parameter shapes × bytes), KV-cache tensor sizes, tied-weight copies |
| **ESTIMATED** | a formula with assumptions | activation peaks, quantized weights, load buffers, stage *times* |
| **UNKNOWN** | not determinable | (no default item is unknown; unknown *worker* memory makes a worker unusable) |

A stage total carries the **weakest** label among its items. Since
activations are always estimated, every stage total is ESTIMATED.
Nothing in this phase is presented as measured memory for a loaded
model: **GPT-2's weights were never loaded**. The estimator works from
the real parameter shapes obtained on the meta device.

## Inputs

- **`Precision`**:
  - `weight_dtype` fp32 | fp16 | bf16.
  - `quantization` none | int8 | int4.
  - `activation_dtype`: default = weight dtype, or fp16 when quantized.
  - `kv_cache_dtype`: default = activation dtype.
  - `quant_group_size`: default 128.
  - `quantize_embeddings` / `quantize_lm_head`: default false.
- **`Workload`**:
  - `batch_size`.
  - `context_length`: the KV-cache length; default is the model maximum, 1024 for GPT-2.
  - `prompt_length`: tokens in one prefill pass; default = context, the worst case.
  - `logits_positions`: all | last.
- **`MemoryPolicy`**:
  - `runtime_overhead_bytes`.
  - `extra_safety_fraction`: default 0.10.
  - `include_kv_cache`, `include_load_temp`.
  - `checkpoint_dtype`: default fp32.

## Formulas

B = batch, T = prompt tokens, L = context length, h = hidden size,
I = MLP inner size, H = heads, a = activation bytes.

| Item | Formula | Label |
|---|---|---|
| Weights, unquantized | Σ numel × bytes(dtype) over the stage's own parameters | CALCULATED |
| Weights, int8 | linear weights: numel × 1 B + out_channels × 2 B (fp16 per-channel scales). Other parameters stay at the activation dtype. | ESTIMATED |
| Weights, int4 | linear weights: ⌈numel/2⌉ + ⌈numel/group⌉ × 4 B (fp16 scale + zero per group) | ESTIMATED |
| Tied-weight copy | if a stage holds `lm_head` but not `wte` (they are the same tensor in GPT-2), the stage needs its own copy: V × h × bytes | CALCULATED |
| KV cache | per block: 2 × B × L × kv_heads × head_dim × bytes(kv dtype) | CALCULATED |
| Activations, block | max(attention phase, MLP phase) × a. Attention phase = 6·B·T·h + 2·B·H·T·L (scores and probabilities materialized). MLP phase = 2·B·T·h + 2·B·T·I. | ESTIMATED |
| Activations, LM head | (B·T·h + B·T_logits·V) × a, where T_logits = T, or 1 with `logits_positions: last` | ESTIMATED |
| Stage activation peak | **max** over the stage's units, not the sum, because inference frees activations layer by layer | ESTIMATED |
| Outgoing buffer | B·T·h·a, on every stage except the last | ESTIMATED |
| Load buffer | largest single tensor × bytes(checkpoint dtype), for the transient copy during load or dtype conversion | ESTIMATED |
| Runtime overhead | default 700 MiB, once per worker | MEASURED (Computer 1) |

**Device vs. host.** On a CPU stage everything is RAM. On a GPU stage,
weights, KV cache, activations and buffers are VRAM, while the runtime
overhead and load buffer are host RAM. Each is checked against its own
capacity.

### Runtime overhead: the measurement
On Computer 1 (2026-10-31; Python 3.13.3, torch 2.10.0+cu126,
transformers 5.20.1), a Python process's RSS was:
- 19 MiB at start;
- **515 MiB** after `import torch, transformers`;
- **698–701 MiB** after loading the GPT-2 config and building the model
  on the meta device. The meta model itself adds about 2–19 MiB.

The default of 700 MiB is that figure. It is specific to this machine
and these library versions, and it is assumed for every worker,
including simulated ones. Override it with `MemoryPolicy(runtime_overhead_bytes=...)`.

## GPT-2 (124M) results

These come from the real config and meta-device shapes, via
`python -m partitioner memory`. They are estimates for one worker
holding the whole model, with batch 1:

| | fp32, ctx 1024 | fp16, ctx 1024 | fp16 + int8, ctx 512, logits last |
|---|---|---|---|
| Weights | 474.70 MiB (CALCULATED) | 237.35 MiB (CALCULATED) | 156.51 MiB (ESTIMATED) |
| KV cache | 72.00 MiB | 36.00 MiB | 18.00 MiB |
| Activation peak | 199.32 MiB (LM-head logits) | 99.66 MiB | 16.50 MiB |
| Load buffer | 147.24 MiB | 147.24 MiB | 147.24 MiB |
| Runtime overhead | 700.00 MiB | 700.00 MiB | 700.00 MiB |
| **Total** | **1.56 GiB** | **1.19 GiB** | **1.01 GiB** |

Per unit at fp32: embeddings 39,383,808 params (150.24 MiB); each
block 7,087,872 params (27.04 MiB, plus 6 MiB of KV cache and a
114 MiB activation peak); final norm 1,536 params; `lm_head` 0 owned
params (tied). 124,439,808 × 4 B = 497,759,232 B = 474.70 MiB exactly.

**The runtime overhead dominates for a model this small.** It is about
44% of the fp32 total (700 of ~1,597 MiB), and every extra stage pays it again. That is why
the optimizer often uses fewer workers than are available for GPT-2.

## Checkpoint facts (read from the real file's header, not downloaded)

- **Contents.** `model.safetensors` stores 160 tensors, all F32: 137,022,720 values,
  548,090,880 bytes.
- **What the extra values are.** 12,582,912 of those values are 12 legacy
  `h.N.attn.bias` causal-mask buffers (1×1×1024×1024 each). They are not
  parameters, and current transformers ignores them.
- **Match with the model.** Every model parameter is present with an
  identical shape, and `lm_head` is absent because it is tied to `wte`.

## Limitations

- **Attention kernel.** Activations assume eager attention.
  `scaled_dot_product_attention` kernels (flash or memory-efficient)
  don't materialize the T×L scores, so real peaks may be lower; the
  estimate errs high.
- **Allocator behaviour.** Allocator fragmentation, the CUDA context
  (often 300–500 MB on NVIDIA) and Python object overhead beyond the
  measured runtime figure are not modelled.
- **Quantization.** Quantized sizes are format-dependent estimates
  (bitsandbytes, GPTQ and AWQ differ). **No quantized runtime exists in
  this project**, so plans set `quantization_runtime_implemented: false`.
- **bf16/fp16 on CPU.** These are estimated for memory only. Their speed
  on CPU is not modelled.
- **Single measurement.** The runtime overhead was measured once, on
  one machine.
