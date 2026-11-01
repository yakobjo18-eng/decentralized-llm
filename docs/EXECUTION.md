# Execution guide (Phase 5): minimal local inference

Phase 5 takes a validated Phase 4 partition plan and runs **real model
inference** for it, in one process, on the machine where you run the
command. It is deliberately small:

- only the built-in test model `builtin:tiny-gpt2-v1` is executable;
- a single forward pass (prefill) over a batch of equal-length token
  sequences, with no KV cache and no autoregressive generation;
- fp32 only.

Code: `model-partitioner/partitioner/execution/`. Tests:
`model-partitioner/tests/test_execution_*.py`.

## Status

| Status | What |
|---|---|
| **Implemented and tested locally** | Plan loading and validation for execution; the tiny model (explicit PyTorch math, deterministic weights); per-stage execution with shape/dtype/finite checks at every boundary; an unpartitioned reference path with an automatic comparison; single-worker real execution; labelled local emulation of multi-stage plans; resource limits checked before allocation; structured results and errors; `python -m partitioner execute`. |
| **Interfaces prepared for future transport** | `StageSpec`, `Activation` / `ActivationMeta`, `StageExecutor`, `ActivationTransport` (`STAGE_INTERFACE_VERSION = 1`). The only transport is `InProcessTransport`, which hands the same tensor object to the next stage. |
| **Not implemented** | Remote tensor transport (no gRPC activation messages, no tensor serialization); physical multi-machine inference; GPT-2 (or any real model) execution; quantized inference; KV cache and token-by-token generation; padding and attention masks. |

## The supported model: `builtin:tiny-gpt2-v1`

| Property | Value |
|---|---|
| Architecture | GPT-2 (`GPT2Config`), planned by the unchanged Phase 4 `GPT2Adapter` |
| Size | 4 layers, hidden 32, 4 heads (head_dim 8), MLP 128, vocab 128, 32 positions |
| Parameters | 56,000 (tied `lm_head` ↔ `wte`), 224,000 B in fp32 |
| Units | `embeddings`, `block.0`–`block.3`, `final_norm`, `lm_head` (7) |
| Weights | `formula-sin-v1`: each value is a fixed formula of the parameter name and element index (no RNG, no download). They are **untrained**, so the logits carry no linguistic meaning. |
| Math | Conv1D layout (`x @ W + b`), pre-LayerNorm, causal eager attention scaled by 1/√head_dim, tanh GELU, no dropout |

Its config is built in, so planning it needs no network access. Any
machine rebuilds bit-identical weights for exactly the components its
stage holds. A stage that has the LM head but not the embeddings gets its
own copy of the tied weight (the plan's `tied_weight_copy` memory item).

### How correctness is verified

| Check | Comparison | Tolerance | Observed (Computer 1) |
|---|---|---|---|
| Partitioned vs. unpartitioned reference | every 1-, 2-, 3- and 7-stage split vs. `reference_forward` | `REFERENCE_ATOL = 1e-6` | 0.0 (bit-identical: same kernels, same shapes) |
| Independent oracle | NumPy float64 GPT-2 forward pass written separately in the tests | 5e-4 absolute | 1.2e-4 |
| Hugging Face cross-check | `transformers.GPT2LMHeadModel` loaded with the same weights (`strict=True`) | 5e-4 absolute | 2.0e-4 |
| Known output | golden logits and next-token ids `[60, 96]` for a fixed input, taken from the NumPy oracle | 5e-4 absolute | passes |
| Causality | changing the last token leaves earlier positions unchanged | 1e-6 | passes |

The two 5e-4 bounds cover fp32 rounding on logits of magnitude about 11
(about 1e-5 relative). Every `execute` run repeats the
partitioned-vs-reference comparison unless you pass
`--no-reference-check`. A mismatch fails the run.

## Modes

| | `real` (`--local WORKER_ID`) | `local-emulation` (`--emulate-stages-locally`) |
|---|---|---|
| Where stages run | this process; every stage must be assigned to this worker on this host | this process, **whatever** worker each stage is assigned to |
| SIMULATED workers | **refused** | allowed, with a warning |
| Plan `executable` | must be `true` | ignored, with a warning |
| Worker records | required; this machine is profiled for real (Phase 3) | optional |
| Multi-stage plans | refused (no transport between workers) | run sequentially, in-process handoffs |
| Result | `execution_mode: local-single-worker`, `assignment_honored: true` | `execution_mode: local-emulation`, `assignment_honored: false`, warning "LOCAL EMULATION … NOT honored … not distributed" |

`distributed` is `false` in every Phase 5 result.

## What makes a plan loadable

The loader checks the plan **as written** and never repairs it. It
collects every problem it finds and returns them as a structured
`rejected` result.

1. The file parses, is a JSON object, and has a supported
   `partition_plan_schema_version`.
2. `inputs_fingerprint` still matches the content. Edited plans are
   refused; re-plan instead.
3. `feasible` is true; otherwise the plan's own infeasibility reasons are reported.
4. `model.model_id` is executable (only `builtin:tiny-gpt2-v1`), and the
   adapter version matches.
5. `precision` is fp32 with no quantization.
6. `validate_plan` passes, with memory recomputed from the plan's own
   settings. This covers ranges, components, workers, profile sources,
   capacity and the config fingerprint (unknown config versions are refused).
7. **Real mode only:**
   - worker records are supplied;
   - no stage is SIMULATED;
   - `executable` is true;
   - every stage is on this process's worker and this host.

The planner sets `executable: true` only for a feasible fp32 plan of an
executable model with exactly one stage on a REAL worker. Every other
plan explains why in `executable_reason`. GPT-2 plans keep
`executable: false`.

## Commands

Run from `model-partitioner/`.

```powershell
# 1. Plan on THIS machine (REAL profile).
python -m partitioner plan --model builtin:tiny-gpt2-v1 --local computer-1 --batch 2 --out tiny.json

# 2. Execute it for real, here.
python -m partitioner execute tiny.json --local computer-1 --tokens "1,2,3,4,5,6,7,8;9,10,11,12,13,14,15,16" --out result.json

# Multi-stage plan from simulated profiles: emulated in one process, labelled.
python -m partitioner plan --model builtin:tiny-gpt2-v1 --simulated ../worker/profiles/simulated/*.yaml --require-all-workers --out sim.json
python -m partitioner execute sim.json --emulate-stages-locally --tokens "5,6,7"

python -m partitioner validate tiny.json --model builtin:tiny-gpt2-v1
```

- `--tokens`: comma-separated ids; `;` separates sequences, which must
  all have the same length.
- `execute` prints the result JSON to stdout and a one-line summary to
  stderr.
- Exit codes: 0 succeeded, 1 rejected or failed, 2 usage error (no mode
  or both modes, unparseable `--tokens`).

**Computer 1 and the RAM margin.** Every stage's memory includes the
**measured** ~700 MiB runtime overhead (torch + transformers). Under the
orchestrator's default Phase 3 margin (10% of RAM, minimum 1 GiB),
Computer 1's usable RAM is usually too small even for the tiny model. On
2026-11-01 step 1 was **INFEASIBLE**: 700.32 MiB needed, 338.94 MiB
effective capacity.

`--ram-margin-percent` and `--ram-margin-min-gib` (with `--local`) set a
smaller margin. The CLI prints a note whenever it differs from the
default. Use the **same** margin flags for `plan` and `execute`, because
`execute` re-profiles this machine and checks that the plan's stated
capacity still fits. A machine whose free RAM has dropped since planning
is refused, which is the intended behaviour.

```powershell
python -m partitioner plan    --model builtin:tiny-gpt2-v1 --local computer-1 --ram-margin-percent 2 --ram-margin-min-gib 0.25 --batch 2 --out tiny.json
python -m partitioner execute tiny.json --local computer-1 --ram-margin-percent 2 --ram-margin-min-gib 0.25 --tokens "1,2,3,4,5,6,7,8;9,10,11,12,13,14,15,16"
```

## Limits (checked before any tensor is allocated)

| Limit | Default | Flag |
|---|---|---|
| Batch size | 8 | `--max-batch` |
| Sequence length | model maximum (32) | `--max-seq` |
| Largest boundary tensor or per-unit activation peak | 64 MiB | `--max-activation-mib` |
| Weight bytes materialized by the run (all stages + reference) | 64 MiB | `--max-weight-mib` |
| Planned workload | batch ≤ plan `workload.batch_size`; tokens ≤ plan `workload.prompt_length` | from the plan |

Input checks, all on the raw lists before any tensor exists:
- integer ids in `[0, vocab)`;
- non-empty sequences of equal length.

Sizes come from the spec:
- boundary tensors: `[batch, tokens, hidden]` or `[batch, tokens, vocab]`
  × bytes, CALCULATED;
- per-unit peaks: from the Phase 4 activation estimator (eager attention),
  ESTIMATED.

Only after every check passes are the token tensor and the stage weights
created. Tests replace the weight and tensor constructors with functions
that fail if called, to prove this.

## The execution result (`execution_result_schema_version` 1)

| Field | Meaning |
|---|---|
| `status` | `succeeded`, `rejected` (nothing ran: plan, input or limits), or `failed` (an error during stage construction, execution or the reference check) |
| `execution_mode`, `distributed`, `distributed_note`, `transport` | how it ran; `distributed` is always false |
| `request_id`, `stage_interface_version`, `executed_on` | hostname, pid and worker id of this process |
| `model` | `model_id`, `config_fingerprint`, `weights_version`, `total_params`, `units` |
| `plan` | provenance: `plan_id` (= `inputs_fingerprint`), schema version, `created_at`, `planner`, `includes_simulated_workers`, `executable` (+ reason), and the plan's own `warnings` and `assumptions` |
| `worker_assignments` | per stage: planned worker, profile source and host, `unit_range`, units, `executed_on_hostname`, `assignment_honored` |
| `limits`, `input`, `preflight` | limits used, input shape and dtype, the sizes that were checked |
| `stages` | per stage: input and output kind/shape/dtype, `elapsed_seconds` |
| `handoffs` | each in-process boundary crossing (kind, shape, dtype, bytes) |
| `output` | `kind` (logits), `shape`, `dtype`, `next_token_ids` (argmax at the last position), min/max; full `logits` with `--include-logits` |
| `reference_check` | `max_abs_diff` vs. the unpartitioned reference, `atol`, `passed` |
| `elapsed` | wall clock of this one run, MEASURED. It includes plan validation and building the model spec. It is not a benchmark. |
| `warnings` | execution warnings (emulation, simulated workers, unchecked workers) |
| `error` | `type`, `message`, `phase`, `stage_index`, `problems` (every rejection reason), `causes` (the exception's cause chain), `traceback` (last frames) |

## The stage interface (for later transport)

```
StageSpec(interface_version, model_id, config_fingerprint, stage_index, n_stages,
          unit_range [start, end), unit_names, components, input_kind, output_kind, activation_dtype)
Activation(meta: ActivationMeta, tensor)
ActivationMeta(interface_version, request_id, kind, shape, dtype, produced_by_stage, positions_start=0)
StageExecutor.forward(Activation) -> Activation
ActivationTransport.send(Activation, to_stage) -> Activation
```

- Kinds:
  - `token_ids` `[B, T]` int64, into stage 0;
  - `hidden_states` `[B, T, hidden]`, between stages;
  - `logits` `[B, T, vocab]`, out of the stage that holds `lm_head`.
- Both the executor and the runner check every field. The runner checks
  each stage's output against what the plan requires, independently of
  the metadata the stage wrote. It rejects:
  - kind, shape, dtype, producer or request-id mismatches;
  - metadata that doesn't describe the tensor;
  - NaN or infinite values.
- `positions_start` is carried explicitly so that a later decode step
  never re-derives positions on a worker. It must be 0 in Phase 5.
- A future remote transport must carry every `ActivationMeta` field
  (plain ints and strings) next to the raw tensor bytes, and bump
  `STAGE_INTERFACE_VERSION` if the contract changes.

## Current limitations

- One model, untrained formula weights, fp32, CPU, a single prefill
  pass, equal-length sequences, no attention mask, no KV cache, no
  generation loop.
- Execution lives in `model-partitioner/` as a library and CLI. The
  worker process (`worker/worker/execution`, `worker/worker/inference`)
  is unchanged, and Phase 3 registration and heartbeats are untouched.
  Hosting stages inside the worker process is later work.
- Stage timings are wall-clock figures from one run. They are not
  benchmarks and are not fed back into the planner.
- Multi-stage execution exists only as labelled single-process
  emulation. It shows that stage boundaries preserve the computation,
  and nothing about network cost or multi-machine behaviour.
