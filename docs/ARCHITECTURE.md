# Decentralized Edge-Compute Pipeline for Large Model Inference
## Phase 0 — Architecture, Risk Analysis, and Implementation Roadmap

Status: DRAFT — awaiting approval before Phase 1 (Foundation) begins.
Date: 2026-10-30

---

## 0. Spec note

`docs/PROJECT_SPEC.pdf` does not exist — this directory and the whole project
tree were empty before this document was created. The working specification
for everything below is the master development prompt supplied in chat. If a
separate PDF spec exists elsewhere, point me to it and I will reconcile this
document against it; otherwise this document treats the chat prompt as the
spec of record.

---

## 1. Verified environment (Computer 1 — this machine)

Collected via PowerShell CIM queries and command probes, not assumed.

| Item | Value |
|---|---|
| OS | Windows 11 Home Single Language, 64-bit, build 10.0.26200 |
| CPU | 13th Gen Intel Core i7-1355U — 10 cores / 12 logical processors, 1.7 GHz base |
| RAM | 15.72 GB total |
| GPU | Intel Iris Xe Graphics (integrated), ~2 GB shared AdapterRAM, driver 32.0.101.6881 |
| NVIDIA / CUDA | Not present. nvidia-smi not found. No discrete GPU. |
| Free disk (C:) | ~207 GB free |
| Python | 3.13.3 (C:\Program Files\Python313) |
| Node.js | v22.19.0, npm 10.9.3 |
| Git | 2.51.0.windows.1 |
| Go | Not installed |
| Docker | Not installed |
| protoc (Protocol Buffers compiler) | Not installed |
| Redis / redis-cli | Not installed |

Conclusion for Computer 1: CPU-only. Iris Xe is an integrated GPU with no
CUDA support; PyTorch cannot use it for acceleration without DirectML, which
is a separate, slower, and less-supported path. Computer 1's worker will run
PyTorch on CPU with float32 (or bfloat16 if AVX-512/AMX paths help — to be
benchmarked, not assumed).

### Blocker: Computers 2 and 3 are unverified

I have a terminal session only on Computer 1. Per the master prompt's own
rule (do not assume any GPU, CUDA capability, or available memory that has
not been verified), I cannot design memory-aware partitioning or decide the
demo model size until I know the real specs of the other two machines. This
is the single most consequential open item — model choice, partition
strategy, and the "requires more memory than any one machine has" demo claim
all depend on it.

Action needed from you: on Computer 2 and Computer 3, run the same probe
and send me the output (or give me remote/SSH/RDP access and I will run it
myself):

```powershell
Write-Output "=== OS ==="; (Get-CimInstance Win32_OperatingSystem).Caption
Write-Output "=== CPU ==="; Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors, MaxClockSpeed | Format-List
Write-Output "=== RAM ==="; "{0:N2} GB" -f ((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
Write-Output "=== GPU ==="; Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM, DriverVersion | Format-List
try { nvidia-smi } catch { Write-Output "No NVIDIA GPU detected" }
Write-Output "=== Disk free (C:) ==="; Get-PSDrive C | Select-Object Used,Free
python --version
```

Until this comes back, all memory budgets and the final demo model size
below are provisional, not committed.

---

## 2. Technology stack — confirmed and adjusted

The master prompt's stack is sound and I am keeping it, with one clarification:

- Orchestrator: Go + REST API. Go must be installed on Computer 1
  (not present yet — Phase 1 task).
- Workers: Python 3.13 + PyTorch + Transformers. Note: PyTorch wheel
  compatibility with Python 3.13 will be verified during Phase 1 setup — if
  there is a gap we will pin a Python 3.11/3.12 venv instead, per the master
  prompt's own mention of Python 3.11.
- Communication: gRPC + Protocol Buffers, binary tensor payloads (never
  JSON for tensors). protoc needs installing in Phase 1.
- Service discovery / heartbeat state: decided (2026-10-30) to defer
  Redis entirely. Phase 1 uses an in-memory registry
  (orchestrator/internal/registry) behind a Registry interface, so a
  Redis-backed implementation can be swapped in later without changing
  any caller (the gRPC server, the REST API). No Docker introduced yet.
- Frontend: React + Next.js + TypeScript + Tailwind + Recharts. Node is
  already installed and current.
- Testing: Go test, Pytest, integration tests.

---

## 3. Model architecture choice — two-tier plan

Rule 13 forbids assuming arbitrary HF models partition generically, and
Rule 5/Section 2 forbid assuming memory not yet verified. So model choice is
split into two tiers instead of one model for everything:

### Tier 1 — Correctness-validation model: GPT-2 (124M, "gpt2")

Used through Phases 4-6 to build and verify the partitioning engine, gRPC
tensor transport, and distributed autoregressive generation against a
single-machine reference, before worrying about scale.

Why GPT-2 specifically:
- Canonical decoder-only Transformer, the architecture this master prompt
  describes end-to-end (embeddings, then N identical blocks, then final
  norm, then LM head).
- Exact, stable module layout in Hugging Face Transformers: wte/wpe
  embeddings, transformer.h[i] blocks (each with ln_1, attn, ln_2, mlp),
  ln_f, lm_head. Easy to write a correct, specific adapter for, not a
  generic "works for any model" claim.
- About 500 MB in float32, downloads in seconds, runs on any CPU, including
  this one. Lets us prove the pipeline is correct cheaply before spending
  hours on a bigger download across three machines.
- Single-machine reference run is trivial to produce for comparing
  greedy-decoded tokens and intermediate hidden states against the
  distributed run.

### Tier 2 — Real distributed-necessity demo model: pending Computer 2/3 specs

The master prompt's actual objective (Section 16) is to show a model too
large for any one machine running across three. With Computer 1 confirmed
at 16 GB RAM / CPU-only, candidates held in reserve, ascending ambition:

| Candidate | Params | fp32 weight memory (approx.) | Fits on 1 machine (16 GB)? |
|---|---|---|---|
| Llama-3.2-1B | 1B | ~4.5 GB | Yes, not a real distributed-necessity demo alone |
| Llama-3.2-3B | 3B | ~13 GB | Marginal on this machine alone, no room for activations/KV-cache at any real context length |
| Qwen2.5-7B / Llama-3.1-8B | 7-8B | ~28-32 GB | No, if Computers 2/3 also have ~16 GB — this is the one that actually demonstrates the project's thesis |

Tier 2 will be finalized once Computer 2/3 RAM is known. If all three
machines turn out to have roughly 16 GB like this one, a 7-8B parameter
model (float32) or a larger model loaded in bfloat16 is the right target:
its full weights alone exceed any single machine's RAM, but three machines'
combined ~48 GB comfortably covers weights plus activations plus KV-cache
with margin. This will not be downloaded or committed to until the hardware
survey is back — doing so earlier would mean guessing at a number the
master prompt explicitly says not to guess at.

---

## 4. Critical technical risks (Section 17 required analysis)

### 4.1 Distributed Transformer execution — the real hard problem

Autoregressive decoding is inherently sequential: token t+1 depends on
token t. In a pipeline-parallel layout, generating one token requires the
activation to cross every worker boundary once (prefill), and then, for
every subsequent token, cross all (W-1) network hops again before the next
token can even start. Per-token latency is approximately:

  compute time summed across each worker's assigned layers
  + (W-1) x (network round-trip time + serialize/deserialize time)

For 3 workers on a LAN (sub-millisecond to low-single-digit-ms round trip),
this overhead is likely small relative to CPU compute time for a model like
GPT-2 or even a 7B model on CPU. But it will not make generation faster than
single-machine execution when the model already fits on one machine — the
point of this architecture is to run models that do not fit on one machine
at all, not to speed up models that do. The benchmarking dashboard will
state this plainly rather than imply distributed equals faster.

Correctness requirements designed in from the start:
- Attention mask and position IDs are threaded through every hop as
  explicit tensor-transport metadata, never re-derived locally (and
  incorrectly) on each worker.
- KV-cache for a given request's assigned layers lives only on the worker
  that owns those layers, keyed by request_id, never shared or
  reconstructed elsewhere.

### 4.2 Heterogeneous CPU/GPU support

Confirmed: Computer 1 has no CUDA device. The worker hardware profiler
(Phase 3) must never assume torch.cuda.is_available() is true, must report
"cpu" as a first-class fully-supported device, and the partitioner treats
GPU-equipped workers as a bonus, not a requirement. Execution dtype defaults
to float32 on CPU workers (fp16 matmul is not well accelerated on most
consumer CPUs); bfloat16 will be benchmarked, not assumed faster.

### 4.3 Windows networking

- gRPC works fine over Windows TCP; no issue there.
- Windows Defender Firewall blocks unsolicited inbound connections by
  default. Each worker's gRPC port and the orchestrator's REST/gRPC ports
  need explicit inbound allow rules (New-NetFirewallRule), documented in the
  Windows deployment guide (Phase 1/10 deliverable). All three machines'
  network profile should ideally be "Private" for the simpler firewall
  path; both the simple (Private network) and explicit-rule (Public
  network) routes will be documented.
- No IPs hardcoded anywhere — orchestrator and worker addresses come from a
  YAML config plus environment variable overrides (worker.yaml, .env), per
  Rule 11.

### 4.4 Model memory requirements and partition safety

The partitioner budgets, per candidate worker and candidate partition:

  required_memory = weight_memory(assigned_layers, dtype)
                   + activation_memory(batch, seq_len, hidden_dim, dtype)
                   + kv_cache_memory(assigned_layers, batch, max_seq_len, num_kv_heads, head_dim, dtype)

against available_memory(worker) x safety_margin (margin starts at 0.8,
tunable). A partition that does not fit is rejected at planning time, never
attempted and left to OOM at runtime.

### 4.5 Tensor transmission

Protobuf message carries: raw tensor bytes (contiguous, via
tensor.numpy().tobytes() — never pickle, for both safety and portability),
explicit shape (repeated int64), explicit dtype (enum, not inferred),
request_id, pipeline_stage_id, and a transmission timestamp for latency
measurement. Deserialization reconstructs via
np.frombuffer(...).reshape(shape).astype(dtype) then torch.from_numpy(...),
verified byte-for-byte against the pre-send tensor in an integration test
(Phase 5).

### 4.6 KV-cache handling

Each worker keeps a dict keyed by request_id, scoped to only the layers it
owns. The orchestrator is responsible for telling workers when a request_id
is complete (success, error, or timeout) so caches get evicted — otherwise
this is a memory leak under concurrent load. Concurrent requests never share
a cache slot; isolation is by request_id, enforced at the worker's
execution-engine layer, not just assumed by convention.

### 4.7 Worker failure recovery — honest limitation up front

If a worker fails mid-generation for a request whose KV-cache lives on that
worker, that cache is gone — it cannot be meaningfully reconstructed on a
replacement worker, because it is a function of that worker's exact
executed layers and weights. Per Rule 7, this will not be glossed over. The
practical, honest recovery design for Phase 8 is:

- New requests: reroute immediately to a healthy worker holding an
  equivalent partition, or redo the partition plan.
- In-flight requests on a failed worker: detected via heartbeat timeout, the
  request is marked failed, logged, and, if chosen, restarted from scratch
  (full prefill again) on a surviving or replacement assignment. This is
  not seamless mid-generation recovery; it is correct-but-expensive
  recovery, and will be labeled exactly that in the dashboard and docs.
- True KV-cache migration/checkpointing across machines is a stretch goal
  to revisit only after Phase 8's basic version works, per Rule 15
  (optimize only after the basic pipeline is correct).

---

## 5. Partitioning algorithm

> **Update (Phase 4, 2026-10-31):** the Phase 4 master prompt asked for
> bottleneck minimization directly, so the dynamic-programming approach
> planned below for Phase 7 was implemented in Phase 4 instead, and the
> greedy fill was never built. See docs/MODEL_PARTITIONING.md. Phase 7
> now means feeding that DP *measured* per-layer and per-link costs.
> The text below is the original Phase 0 plan, kept for history.

Phase 4 (initial): greedy sequential fill. Walk the model's layers in order,
accumulate onto the current worker (in pipeline order) until adding the
next layer would exceed that worker's available_memory x margin, then move
to the next worker. Reject the whole plan if any single layer cannot fit on
any worker at all.

Phase 7 (latency-aware upgrade): explicitly not a shortest-path problem — a
Transformer's layers are a fixed sequential chain, not a graph to route
through. It is a chain-partitioning / minimize-the-bottleneck-stage
problem: given fixed layer order and fixed worker order, choose split
points to minimize the slowest stage, where each stage's cost is
compute_time(its layers) + transmit_time(output tensor to next hop),
subject to the memory feasibility check above. This will be implemented as
dynamic programming over (layer_index, worker_index) -> minimum possible
worst-stage-time, the standard justified approach for this exact problem
shape (equivalent to classic assembly-line/chain-partition balancing, not
generic graph routing).

---

## 6. Repository structure

Adopting the master prompt's structure as-is (Section 10) — it is already
well factored and there is no architectural reason to deviate. This will be
created in Phase 1, not yet.

---

## 7. Phase status

See docs/DEVELOPMENT_STATUS.md for the live tracker. Current state: Phase 0
complete (this document). Phases 1-10 not started.

---

## 8. What has deliberately not been done yet

Per the master prompt's own instruction (do not start generating the entire
application immediately, wait for approval before implementing Phase 1):

- No git init, no repo scaffolding, no go.mod/package.json/proto files
  created yet.
- No dependencies installed (Go, Docker, protoc, Redis all still missing on
  this machine and will be installed as Phase 1 tasks once approved).
- No model downloaded yet.

## 9. Decisions (2026-10-30, approving Phase 0 / authorizing Phase 1)

1. Computer 2 / Computer 3 hardware — not yet collected. Explicitly does
   NOT block Phase 1 or Phase 2; must be collected before Phase 4
   (model partitioning), where guessing memory/GPU specs would violate
   Rule 5 directly.
2. Redis — deferred. In-memory registry behind a clean interface for
   now (see section 2 above); no Docker introduced yet.
3. Network — all three computers are on the same private Windows LAN.
   Configurable IPs/ports (never hardcoded) and documented Windows
   Firewall rules; see docs/NETWORKING.md.
4. docs/PROJECT_SPEC.pdf — to be placed by the user; will be read and
   reconciled against this document once present. Not present as of
   Phase 1.

See docs/DEVELOPMENT_STATUS.md for what Phase 1 actually built and its
verified test results.
