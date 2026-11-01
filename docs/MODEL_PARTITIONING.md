# Model Partitioning Engine (Phase 4)

`model-partitioner/` assigns **contiguous ranges of Transformer layers**
to heterogeneous workers, so that the **slowest pipeline stage** is as
fast as possible, within each worker's available memory. It inspects a
real Hugging Face model without loading its weights, and produces a
validated, versioned plan.

It does **not** move tensors or run inference. Phase 5 added minimal
local execution of plans for the built-in test model only (see
[EXECUTION.md](EXECUTION.md)). Every GPT-2 plan still says
`executable: false`.

Related: [MEMORY_ESTIMATION.md](MEMORY_ESTIMATION.md) ·
[PARTITION_PLAN_SCHEMA.md](PARTITION_PLAN_SCHEMA.md) ·
[SIMULATED_HARDWARE.md](SIMULATED_HARDWARE.md)

## Pipeline

```
model id ──► config.json ──► GPT-2 adapter ──► meta-device model ──► ModelSpec
             (cache/Hub)     (validates)       (real shapes, 0 B)    ordered units + components
                                                                        │
workers ──► WorkerResources (Phase 3 shape) ──► cost.resolve_target ────┤ device, throughput, capacity
  orchestrator /api/resources │ simulated YAML │ this machine            │
                                                                        ▼
                     MemoryEstimator.stage(i, j) ─┐       optimizer.solve (DP) × worker orders
                     unit FLOPs ──────────────────┴──►   stage cost(k, i, j) or "doesn't fit"
                                                                        ▼
                                              plan dict ──► validate_plan ──► returned / saved
```

## 1. Architecture inspection

- **Adapter interface** (`partitioner/adapters/base.py`): `validate_config`,
  `build_meta_model`, `describe → ModelSpec`. Everything downstream is
  architecture-neutral. It only sees **units**, the chain the
  boundaries fall between, made of **components**, the real module paths.
- **GPT-2 adapter** (the only one): maps `transformer.wte`/`wpe`/`drop` to
  `embeddings`, `transformer.h[i]` to `block.i`, `transformer.ln_f` to
  `final_norm`, and `lm_head` to `lm_head`. It finds tied weights by
  tensor identity, not by trusting the config flag. It checks every
  parameter's shape against the config, and **refuses** a model with an
  unmapped module, a missing or extra block parameter, or a block count
  that disagrees with `n_layer`.
- **Other architectures are rejected**, not guessed: `get_adapter`
  raises for any `model_type` without an adapter. Adding Llama-style
  models means writing a new adapter (embed_tokens / layers / norm /
  lm_head, with grouped-query attention via `n_kv_heads`).
- **No hardcoded layer count**: the tests plan a synthetic 4-layer GPT-2
  config end to end.

**Actual inspection of `gpt2`** (Computer 1, 2026-10-31; transformers 5.20.1):

| | |
|---|---|
| Layers / hidden / heads / head_dim | 12 / 768 / 12 / 64 |
| MLP inner / vocab / positions | 3072 / 50,257 / 1,024 |
| Activation | `gelu_new`; LayerNorm eps 1e-5 |
| Parameters | **124,439,808**, which matches the independent formula V·h + P·h + L·(12h² + 13h) + 2h |
| Tied weights | `lm_head.weight` is `transformer.wte.weight` (38,597,376 params) |
| Memory cost of inspecting | meta model +2 to +19 MiB RSS; no weight file downloaded |
| Checkpoint header | verified: 160 F32 tensors, 548,090,880 B. Every parameter is present with its shape. The 12 extra tensors are legacy causal masks (`h.N.attn.bias`), not parameters. |

## 2. Worker resource model

The planner uses the Phase 3 **resources** view: available RAM minus
the Phase 3 margin (max of 10% of total and 1 GiB), and dedicated VRAM
minus usage and its own margin.

- **Sources.** `from_orchestrator(url)` reads a live cluster;
  `from_simulated_profile(path)` and `from_local_machine(id)` work
  offline. The offline formula equals the Go orchestrator's
  (`test_parity_with_live_orchestrator` runs both).
- **Never total RAM.** Usable memory is *available* RAM after margins.
  The planner then keeps another `extra_safety_fraction` (10%) in reserve.
- **Device choice.** A worker runs on a GPU only if that GPU is
  compute-usable (a working backend targets its vendor), its usable VRAM
  is known, **and** a throughput is known (an operator `gpu_gflops`
  override; Phase 3 has no GPU benchmark). Otherwise it runs on the CPU,
  with a warning. A CPU-only worker never receives a GPU memory
  requirement.
- **Conservative on missing data.** A worker is excluded, with the
  reason recorded, when it is offline, its backend probe is pending, its
  available RAM is unknown, it has no CPU backend, its throughput is
  unknowable, or its profile source is UNKNOWN.
- **Shared hosts.** Several REAL workers with the same hostname split
  its usable RAM equally, with a warning. Local processes are not
  independent machines.
- **Utilization.** Throughput is scaled by the idle fraction
  (1 − util%), with a floor of 10%.

## 3. Cost model

- **Formula.** Stage time = calculated FLOPs ÷ effective throughput, plus
  the output transfer time if a link bandwidth is given.
- **FLOPs per unit** come from the real shapes: 2 × tokens × (linear
  weights), plus attention 4·B·T·L·H·d, and 2·B·V·h for the LM head.
- **Cost basis.** The default is one **decode** step (one new token per
  sequence over a full context); `prefill` is the alternative.
- **Throughput, in order of preference:**
  1. operator override;
  2. the Phase 3 matmul micro-benchmark. It is MEASURED but noisy: 31 to
     294 GFLOPS across runs on Computer 1. `--throughput estimated`
     skips it.
  3. cores × max MHz × 8 FLOPs/cycle, an assumed ~25% of AVX2 peak.

**All seconds are ESTIMATED.** The optimizer needs relative costs to
be right; the absolute numbers are not predictions.

## 4. Partitioning algorithm

**Problem.**
- Units 0…U−1 are a fixed chain; workers are taken in a pipeline order.
- Worker k receives a contiguous `[s_k, e_k)`. The ranges tile 0…U, and a
  range may be empty, meaning that worker is unused.
- Cost c(k, i, j) is the stage time, or "doesn't fit" if
  `MemoryEstimator.stage(i, j)` exceeds that worker's capacity.
- **Objective:** minimize max_k c(k, s_k, e_k), the bottleneck. In a
  pipeline, the slowest stage limits throughput. The goal is **not** equal
  layer counts. With GPT-2, for example, the LM head (a 50,257 × 768
  matmul, 77.2 MFLOPs) costs about as much as 4.5 blocks (17.3 MFLOPs
  each) in a decode step.

**Method: exact dynamic programming** (`partitioner/optimizer.py`):
1. B[k][j] = min over i ≤ j of max(B[k−1][i], c(k−1, i, j)), with B[0][0] = 0. The optimal bottleneck is B* = B[m][U].
2. Second pass: minimize the **sum** of stage costs, subject to every
   stage ≤ B*, breaking ties toward the earliest split. This pass is
   exact because the constraint is per-stage. A single pass over
   (max, sum) pairs would not be, because that objective isn't
   decomposable.

**Worker order.**
- For up to 6 workers (`permutation_limit`), every order is tried
  (6! = 720 DP runs). Results are compared by bottleneck, then total
  time, then input order.
- With more workers, the worker-id order is used, with a warning.
- Inputs are sorted by worker id first, so a shuffled input list gives
  the identical plan (tested).

**Complexity.**
- O(m·U²) cost evaluations per order. Stage memory is cached per (i, j),
  because it doesn't depend on the worker.
- For GPT-2 small (U = 15) with 4 workers, all 24 orders take well under
  a second.

**Why not Dijkstra or shortest paths:** there is no routing choice. Layer
order and worker order are fixed, and the objective is a max over
stages, not an additive path length. This is the classic
linear-partition (chain-to-chain mapping) problem, which DP solves
exactly. *Correctness check:* the DP matches an exhaustive brute-force
oracle on 1,500 seeded random instances, including memory-infeasible
ranges, and its secondary objective is checked exhaustively on 200
more.

**Structural consequence worth knowing.** The first stage always holds
the embeddings (150 MiB fp32, plus a 147 MiB load buffer). The last
always holds the LM head (a 199 MiB logits peak, plus a 147 MiB tied
copy of `wte` when the embeddings are elsewhere). So a small-memory
worker can only help in a **middle** position, and that needs at least
3 workers (`test_end_stages_are_heavier_than_middle_stages`).

## 5. Explanations

Each stage explains:
- what it runs;
- the memory needed vs. capacity, with provenance;
- its time, split into compute and transfer, and how close it is to the bottleneck;
- tied-weight copies;
- **boundary sensitivity**: the bottleneck that would result from moving
  each boundary one unit either way, or which worker would then run out
  of memory.

Unused workers say whether they were *too small for any stage* or
*eligible but not needed*, and include the memory figures behind that.
Infeasible plans report the whole-model requirement against the largest
and combined capacities, the hardest single unit, every unusable worker
with its reason, and options to try.

## 6. Example plans (actual output, 2026-10-31)

Files: `model-partitioner/examples/`. In all of them the **REAL** worker
is Computer 1, profiled live, and every **SIMULATED** worker is a
Phase 3 example profile. **None of them describe Computers 2 or 3.**

**(a) Computer 1 alone, REAL. Result: INFEASIBLE.**
- *Requirement:* GPT-2 fp32 at context 1024 needs an estimated 1.56 GiB.
- *What Computer 1 had:* 1.45–1.47 GiB available during these runs, with
  the desktop using about 84% of its 15.72 GiB. After the Phase 3 margin
  (1.57 GiB), that leaves **0 B usable**.
- *Two compounding causes:* the planner's own process (~700 MiB of
  torch + transformers) was resident while it measured the machine it
  was planning for, and the plan warns about this. Even at the 2.5 GiB
  available measured without the planner running, effective capacity
  would be about 0.84 GiB, still short.

So under its current load, Computer 1 cannot host GPT-2 within these
safety margins. This is a measured result, not a defect. Freeing memory,
or lowering `capability.ram_safety_margin_*` /
`--extra-safety`, changes the answer.

**(b) Computer 1 + simulated A, B, C, with B's GPU at an assumed 5,000 GFLOPS (operator override):**

| Stage | Worker | Units | Device | Memory / capacity | Est. time |
|---|---|---|---|---|---|
| 0 | profile-c-high-memory-worker (SIM) | embeddings, block 0 | cpu | 1.12 GiB / 41.04 GiB | 0.047 ms |
| 1 | profile-b-gpu-worker (SIM) | blocks 1–11, final_norm, lm_head | gpu (cuda) | 710 MiB VRAM / 5.94 GiB; 847 MiB host | **0.055 ms (bottleneck)** |

- **Why this split.** The GPU stage is roughly 13× faster than any CPU
  here, so it takes almost everything. Stage 0 keeps the embeddings,
  plus exactly one block, because moving block 0 to the GPU raises the
  bottleneck by 6.47%, and moving block 1 off it raises it by 72%.
- **Unused workers.** Computer 1 is "too little memory for any stage"
  (0 B usable). Profile A is "eligible but not needed".

**(c) Simulated A, B, C, all on CPU (no GPU throughput given), every worker required, 100 Mbit/s link:**

| Stage | Worker | Blocks | Est. time (compute + transfer) |
|---|---|---|---|
| 0 | profile-a-cpu-worker | embeddings + 0–2 | 0.502 ms (0.256 + 0.246) |
| 1 | profile-b-gpu-worker (CPU path) | 3–5 | **0.526 ms (0.280 + 0.246), bottleneck** |
| 2 | profile-c-high-memory-worker | 6–11, final_norm, lm_head | 0.496 ms |

Layer counts are deliberately unequal (3 / 3 / 6, plus the end units).
C has 1.8–2× the estimated throughput of A or B and holds the expensive LM head, and the
three stage times end up within 6% of each other.

## 7. How to use

```powershell
cd "C:\Users\Yakob\OneDrive\Documents\Programs\decentralized llm\model-partitioner"
python -m partitioner inspect --model gpt2
python -m partitioner memory  --model gpt2 --dtype fp16 --context 512
python -m partitioner plan    --model gpt2 --simulated ..\worker\profiles\simulated\profile_a_cpu_worker.yaml `
                                ..\worker\profiles\simulated\profile_c_high_memory_worker.yaml --out plan.json
python -m partitioner plan    --model gpt2 --local computer-1                     # profile this machine (REAL)
python -m partitioner plan    --model gpt2 --orchestrator http://<COMPUTER-1-IP>:8080   # live cluster
python -m partitioner validate plan.json --model gpt2
```

Other options:
- **Precision and workload:** `--dtype fp32|fp16|bf16`, `--quant none|int8|int4`
  (memory estimate only), `--batch`, `--context`, `--prompt`,
  `--cost-mode`, `--logits last`.
- **Worker selection:** `--no-simulated`, `--require-all-workers`.
- **Throughput:** `--cpu-gflops ID=N`, `--gpu-gflops ID=N`,
  `--throughput estimated`.
- **Margins and link:** `--extra-safety F`, `--link-bandwidth-mbps M`.

Exit codes: 0 = feasible, 1 = infeasible (or invalid, for `validate`),
2 = usage error.

**Tests:** `cd model-partitioner; python -m pytest` runs 113 tests in
about 30–100 s. Some of them depend on Computer 1 itself, the network,
or the orchestrator binary:

| Test | Runs against | Takes |
|---|---|---|
| real GPT-2 inspection | cached config, or the Hub | — |
| checkpoint-header test | network; skips honestly if unreachable | — |
| orchestrator-parity test | `orchestrator.exe` | — |
| `from_local_machine` | profiles this machine | ~10 s |

## 8. Limitations

- **One architecture.** Only GPT-2 is supported. Other models are
  rejected until an adapter exists.
- **Cost model.**
  - It is compute-only: CPU decode is often memory-bandwidth-bound, and
    elementwise ops, Python overhead and cache effects are ignored.
  - Throughput comes from a noisy micro-benchmark or a core-count
    assumption, and GPU throughput must be supplied by hand.
- **Transfer time** is modelled only when a link bandwidth is given.
  Real LAN latency between the machines is unmeasured (Phase 7).
- **Memory.** Activation estimates assume eager attention (they err
  high). Quantized sizes are format-dependent estimates, and **no
  quantized runtime exists**. The runtime overhead was measured once, on
  Computer 1.
- **Worker-order search** is exhaustive only up to 6 workers.
- **Local measurement skew.** Planning on the same machine you profile
  lowers its measured available RAM, and the plan warns about this.
- **Simulated hardware.** Simulated plans describe hypothetical hardware
  and say so. No plan has been made for Computers 2 or 3, whose hardware
  is unknown.
- **Nothing executes.** The plans are inputs for Phases 5–6, which have
  not been built.
