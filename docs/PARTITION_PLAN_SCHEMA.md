# Partition Plan Schema — version 1

A partition plan is a JSON document produced by
`partitioner.create_plan()` / `python -m partitioner plan`. It is plain
JSON so that later phases, in any language, can consume it. Every plan
returned by the planner has already passed the validator
(`partitioner/validate.py`). Real example files are in
`model-partitioner/examples/`.

## Versioning
- `partition_plan_schema_version` is an integer, currently **1**. The
  validator rejects versions it doesn't support.
- New optional fields may be added without a version bump. Readers must
  ignore unknown fields.
- Renaming or removing a field, or changing a field's meaning, bumps the
  version.
- `planner.algorithm` (`dp-min-bottleneck-v1`) changes when the
  optimizer's behaviour changes, independently of the schema.

## Top level

| Field | Type | Meaning |
|---|---|---|
| `partition_plan_schema_version` | int | 1 |
| `created_at` | ISO-8601 UTC | when the plan was made (excluded from the fingerprint) |
| `planner` | object | `name`, `version`, `algorithm` |
| `feasible` | bool | false → `stages` is empty and `infeasibility.reasons` explains why |
| `objective` | object | `type` = `minimize_max_stage_time`; `cost_basis`; `bottleneck_seconds`, `total_seconds` (null if infeasible); `seconds_provenance` = `ESTIMATED`; `worker_order_search` |
| `model` | object | see below |
| `precision` | object | `weight_dtype`, `quantization`, `activation_dtype`, `kv_cache_dtype`, `quant_group_size`, `quantize_embeddings`, `quantize_lm_head`, `quantization_runtime_implemented` (always **false** when quantized: memory estimate only) |
| `workload` | object | `batch_size`, `context_length`, `prompt_length`, `cost_mode`, `logits_positions` |
| `memory_policy` | object | `runtime_overhead_bytes` (+ basis, provenance), `extra_safety_fraction`, `include_kv_cache`, `include_load_temp`, `checkpoint_dtype` |
| `cost_model` | object | throughput assumptions and the inter-stage link (null = not modelled) |
| `workers_considered` | array | every input worker: `worker_id`, `profile_source`, `hostname`, `simulated_profile_name`, `origin` |
| `stages` | array | pipeline order (see below) |
| `unused_workers` | array | `worker_id`, `profile_source`, `reason`, `ram_total_bytes`, `ram_available_bytes`, `ram_usable_bytes` |
| `includes_simulated_workers` | bool | true if ANY stage runs on a SIMULATED worker |
| `executable` | bool | Since Phase 5: **true only** for a feasible fp32, unquantized plan of an executable model (only `builtin:tiny-gpt2-v1`) with exactly **one stage on a REAL worker**. Everything else, including every GPT-2 plan, is false. `executable_reason` says why. The execution loader re-checks all of this itself (docs/EXECUTION.md). The field's meaning ("a runtime in this project can execute this plan as written") is unchanged; a runtime now exists for some plans. So there is no version bump, and Phase 4 plans (always false) remain correct. |
| `warnings`, `assumptions` | string arrays | everything a reader must know to trust the numbers |
| `infeasibility` | object or null | `reasons`: total vs. capacity, hardest single unit, unusable workers, options |
| `inputs_fingerprint` | sha256 hex | over everything except `created_at` and itself. Identical inputs give an identical fingerprint. |

## `model`

| Field | Meaning |
|---|---|
| `model_id`, `architecture`, `adapter` | e.g. `gpt2`, `gpt2`, `gpt2 v1` |
| `config` | the architecture-relevant config fields |
| `config_fingerprint` | sha256 of the full config. The validator refuses a plan for a different config. |
| `n_layers`, `hidden_size`, `intermediate_size`, `n_heads`, `n_kv_heads`, `head_dim`, `vocab_size`, `max_positions`, `total_params` | from the inspected model |
| `tied_weights` | `[[tied_param, owner_param], ...]` |
| `units` | the ordered chain: `index`, `name`, `kind`, `components`, `block_index` |
| `unit_memory` | the per-unit memory table used for planning |
| `checkpoint` | the safetensors-header cross-check (`status`, dtypes, stored values and bytes, extra tensors) |

For GPT-2 the units are `embeddings` (= `transformer.wte`, `transformer.wpe`,
`transformer.drop`), `block.0` … `block.11` (= `transformer.h.N`),
`final_norm` (= `transformer.ln_f`), and `lm_head`.

## `stages[]`

| Field | Meaning |
|---|---|
| `stage_index` | 0 .. k-1, in pipeline order |
| `worker_id`, `profile_source`, `simulated_profile_name`, `hostname` | which worker, and whether it is REAL or SIMULATED |
| `unit_range` | `[start, end)`: **the exact boundaries**, as unit indices |
| `layer_range` | `{first_block, last_block}` (inclusive Transformer block indices), or null when the stage holds no blocks |
| `units`, `components` | the names in model order (the components are the module paths a runtime will load) |
| `execution` | `device` (cpu/gpu), `backend`, `gpu_id`, `effective_gflops`, `throughput_provenance`, `throughput_basis`, `utilization_factor`, capacities, `ram_share` |
| `memory` | `device_bytes`, `host_bytes`, `total_bytes`, `provenance`, `items[]` (name, bytes, provenance, basis) |
| `memory_fits` | true (always, for a validated plan) |
| `available_memory` | the worker's `ram_available_bytes` / `ram_usable_bytes` / Phase 3 margin, and the effective `device_capacity_bytes` / `host_capacity_bytes` the planner allowed |
| `estimated_seconds` | compute + transfer for one step (ESTIMATED) |
| `estimated_compute_seconds`, `estimated_transfer_seconds` | the two parts |
| `estimated_flops` | calculated FLOPs for one step |
| `is_bottleneck` | true for the slowest stage(s) |
| `explanation` | human-readable reasons, including boundary sensitivity (what moving each boundary by one unit would do) |

## Invariants (checked by the validator)

1. **Coverage, order, uniqueness.**
   - `unit_range`s are non-empty and contiguous, start at 0 and end at
     `len(model.units)`.
   - So every unit is assigned, none is assigned twice, and the order is preserved.
2. **Stage contents match the model.** Each stage's `units`,
   `components` and `layer_range` exactly match its `unit_range`, and
   every model component appears exactly once across all stages.
3. **Workers.**
   - Each worker holds at most one stage.
   - Every stage worker exists in the given worker records.
   - No worker is both a stage worker and listed as unused.
4. **Source labelling is truthful.** `profile_source` matches the
   worker record, and `includes_simulated_workers` matches the stages.
5. **Memory.**
   - Requirements fit the stated capacities: device + host ≤ capacity on
     a CPU stage, or each within its own capacity on a GPU stage.
   - Stated capacity ≤ the worker's usable RAM.
   - With an estimator, the stated requirements must equal a
     recomputation, so an understated plan is rejected.
6. **Model.** `config_fingerprint`, `architecture`, `n_layers` and the
   unit list match the model being validated against.
7. **Feasibility.** An infeasible plan has no stages and gives reasons.

## Integration points for later phases

- **Phase 5 (tensor transport).** Stage boundaries are `unit_range`
  edges. The tensor crossing a boundary has the shape `[batch, tokens,
  hidden_size]`, and its size is the `outgoing_hidden_state_buffer` item.
- **Phase 6 (distributed inference).** A worker loads exactly its
  `components`. The last stage must also materialize any
  `tied_weight_copy`.
- **Phase 7 (scheduling).** Swap in measured per-layer timings through
  `WorkerOverride` / `CostModel`, and the link model through
  `link_bandwidth_bytes_per_s`. The schema stays the same.
- **Orchestrator.** The planner reads `GET /api/resources` via
  `from_orchestrator(url)`. Nothing in the orchestrator depends on the
  planner yet.
