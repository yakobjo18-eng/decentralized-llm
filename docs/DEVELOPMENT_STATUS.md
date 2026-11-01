# Development Status

Last updated: 2026-11-01 (Phase 8 -- two-machine LAN test readiness; physical test PENDING)

## Completed phases

### Phase 0 — Architecture
See ARCHITECTURE.md. Environment profiled for Computer 1 only.
Two-tier model strategy set (GPT-2 for correctness validation, larger
model for the real demo pending hardware data from Computers 2/3).

### Phase 1 — Foundation
Decisions from approval: in-memory registry (no Redis/Docker yet,
behind a swappable interface), private LAN with documented firewall
rules, don't block on Computer 2/3 hardware.

Built, all real and verified (not stubs returning fake data):

- **Go toolchain installed** (Go 1.27.0 via winget), `protoc-gen-go`
  and `protoc-gen-go-grpc` installed.
- **Protocol Buffer definitions** (`shared/proto/common.proto`,
  `registry.proto`, `health.proto`), compiled to both Go
  (`orchestrator/internal/proto/`) and Python
  (`worker/worker/generated/`).
- **Go orchestrator** (`orchestrator/`):
  - `internal/config` — YAML + env var config loader.
  - `internal/registry` — in-memory `Registry` interface
    implementation (Register/Heartbeat/Deregister/Get/List).
  - `internal/grpcserver` — gRPC `WorkerRegistry` service wired to the
    registry.
  - `internal/api` — REST API: `GET /health`, `GET /api/workers`, with
    CORS for the frontend dev server.
  - `cmd/orchestrator/main.go` — wires it all together, graceful
    shutdown on Ctrl+C.
  - Placeholder packages for `scheduler`, `routing`, `pipeline`,
    `monitoring` (doc comments only, explicitly marked not implemented,
    for later phases).
- **Python worker** (`worker/`):
  - `config/loader.py` — mirrors the Go config pattern.
  - `hardware/profiler.py` — minimal, real hardware snapshot (CPU
    count via `os.cpu_count()`, RAM via `psutil`, GPU presence via an
    actual `nvidia-smi` probe — correctly reports no GPU on Computer 1,
    never assumes one).
  - `communication/registry_client.py` — gRPC client for
    Register/Heartbeat/Deregister.
  - `communication/health_server.py` — gRPC Health service the
    orchestrator (or anyone) can query directly on this worker.
  - `main.py` — `python -m worker.main --config worker.yaml`: loads
    config, starts its Health service, registers, heartbeats every N
    seconds, deregisters on shutdown.
  - Placeholder packages for `inference`, `execution`, `monitoring`
    (not implemented, marked for Phase 4/6/3 respectively).
- **Frontend skeleton** (`frontend/`): Next.js 16.4.3 + TypeScript +
  Tailwind, one connectivity-check page that polls the orchestrator's
  real REST API every 3s and shows actually-registered workers (or an
  honest "unreachable" state with no orchestrator running) — not the
  full 7-page dashboard, which is Phase 9.
- **Configuration management**: `.env.example` (root),
  `orchestrator/orchestrator.example.yaml`,
  `worker/worker.example.yaml`, `frontend/.env.example`. No IP
  addresses hardcoded anywhere.
- **Tests**: 19 Go tests (`go test ./...`), 9 Python tests
  (`pytest`), both passing — see "Test results" below for the actual
  run output.
- **Local smoke test** (`scripts/smoke_test_phase1.ps1`): builds and
  runs the real `orchestrator.exe`, runs the real
  `python -m worker.main`, confirms via the REST API that the worker
  registered with its real hardware data and that heartbeats are
  flowing, then shuts both down. **Passed** (see below).

Deliberately NOT done in Phase 1 (by design, not oversight):
- No model loading, partitioning, or tensor transport (Phases 4-6).
- No heartbeat-timeout detection / marking a worker offline — the RPC
  round-trip works, but the orchestrator does not yet act on a missed
  heartbeat (Phase 2/8). **Done in Phase 2, see below.**
- No multi-worker (2-3 concurrent workers) testing yet (Phase 2).
  **Done in Phase 2, see below.**
- No persistence across orchestrator restarts (would come with a
  Redis-backed `Registry` implementation, not planned yet). Still
  true after Phase 2.
- Full dashboard pages 2-7 (Phase 9). Still true after Phase 2.

### Phase 2 — Worker Registration and Multi-Worker Management

Preserved all Phase 1 functionality; one Phase 1 test
(`TestRegister_SameIDTwiceRefreshesNotDuplicates`) was *updated*, not
removed, because it originally exercised silently overwriting a
worker's address on re-registration — exactly the behavior this
phase's objective 1 asked to eliminate. See the comment left in that
test explaining why. No other Phase 1 file's behavior changed.

Built, all real and verified:

- **Worker identity / duplicate prevention**
  (`orchestrator/internal/registry/registry.go`): `Register()` now
  rejects a registration for a worker ID that is currently online
  (heartbeat seen within `heartbeat_timeout_seconds`) if it comes from
  a *different* address than the existing registration
  (`ErrWorkerIDConflict`). Re-registering from the *same* address (a
  fast restart) or registering a worker ID that has gone offline (a
  genuine reconnect, possibly from a new address/port) is still
  allowed. Hostname/hardware metadata support was already in place
  from Phase 1 and is unchanged.
- **Heartbeat / staleness system**: `InMemoryRegistry` now takes a
  configurable `heartbeatTimeout` (`NewInMemoryRegistryWithTimeout`;
  `NewInMemoryRegistry()` keeps working unchanged, using a 15s
  default). `Online` is computed fresh on every `Get()`/`List()` call
  against `time.Now()` — a derived read-time property, not a value
  mutated by a separate background sweep, so there's no second
  goroutine racing against `Register`/`Heartbeat`/`Deregister`. An
  offline worker stays listed (`online: false`), it is not deleted;
  only `Deregister()` removes an entry.
- **Orchestrator config / REST API**: new
  `heartbeat_timeout_seconds` (YAML) /
  `ORCHESTRATOR_HEARTBEAT_TIMEOUT_SECONDS` (env), default 15s.
  `GET /api/workers` now returns `online` and
  `last_heartbeat_age_seconds` per worker; `GET /health` now returns
  `online_workers` alongside `registered_workers`.
- **Worker CLI / reconnection** (`worker/worker/main.py`):
  - New `_register_with_retry`: the initial connection to the
    orchestrator now retries with backoff (2s → up to 10s) instead of
    crashing the process if the orchestrator isn't up yet. A
    rejected-but-reachable response (e.g. a duplicate-ID conflict) is
    NOT retried — retrying an application-level rejection would just
    get the same rejection again, so it's returned immediately with a
    clear log message explaining the likely cause.
  - Signal handling moved before the connection attempt, so Ctrl+C
    works even while still retrying to connect, not just once the
    heartbeat loop has started.
  - Added `SIGBREAK` handling (Windows), so a test harness or operator
    can gracefully stop one specific background worker process via
    `os.kill(pid, signal.CTRL_BREAK_EVENT)` (started with
    `CREATE_NEW_PROCESS_GROUP`) without force-killing it. This is what
    let Phase 2 actually test the deregister-on-shutdown path for the
    first time (Phase 1's smoke test could only force-kill, which
    never exercises it).
- **Tests added** (33 Go tests total, up from 19; 14 Python tests
  total, up from 9):
  - Go: heartbeat expiry, heartbeat refresh before timeout, conflict
    rejection, same-address re-registration, reconnection after going
    offline, concurrent registration of distinct IDs (50 goroutines),
    concurrent *conflicting* registration (exactly one must win),
    concurrent heartbeats+List under sustained load, REST reflecting
    offline status and the online-worker count, gRPC-layer conflict
    rejection.
  - Python: retry-with-backoff unit tests (succeeds after transient
    failures, stops promptly on shutdown request, does not retry an
    application-level rejection), plus two **real** integration tests
    that build on the actual `orchestrator.exe` and real
    `python -m worker.main` subprocesses:
    `test_graceful_shutdown_integration.py` (CTRL_BREAK_EVENT →
    confirms deregistration via the live REST API) and
    `test_three_worker_integration.py` (the full objective 5 scenario:
    three real worker processes register, one is hard-killed to
    simulate an unexpected disconnection, it's confirmed offline while
    the other two stay online, then it's restarted and confirmed back
    online).
  - `scripts/multi_worker_test.py`: an interactive (non-pytest) runner
    for watching the same scenario manually, with PIDs printed so you
    can kill one yourself in another terminal.

Known limitation, disclosed rather than worked around: `go test -race`
could not be run — this machine has no 64-bit C toolchain, which cgo
(required by the race detector) needs, and installing one was outside
this phase's scope without being asked. Concurrency correctness is
therefore argued from the design (every registry mutation holds a
single `sync.RWMutex` for its whole duration) and verified
functionally (the concurrent-registration and
concurrent-heartbeats-and-List tests above, run and passing under the
plain, non-race test runner), not via the race detector itself.

**Local multi-process testing vs. physical multi-machine testing —
reported separately, as instructed:**
- **Local multi-process (Computer 1 only): done and passing.** Three
  real OS processes, three real worker identities, over loopback.
- **Physical multi-machine (Computer 1 + 2 + 3): NOT done.** No
  worker has registered from a non-loopback address in any test. See
  docs/NETWORKING.md's new "What Phase 2 has NOT verified" section for
  the specific gaps (untested address-advertisement path, unapplied
  firewall rules, unmeasured real LAN latency vs. the chosen timeout
  default).

### Phase 3 — Hardware Profiling and Capability Detection (first pass, schema 2)

*This first pass is kept as written for history. The second pass
below (schema 3) supersedes parts of it, noted there. In particular,
Iris Xe's `memory_total_bytes` is no longer reported, the default
refresh interval is now 300 s, and GPU utilization is implemented.*

Preserved all Phase 1/2 functionality. `HardwareSnapshot`
(`worker/worker/hardware/profiler.py`) kept every Phase 1/2 field name,
type, and position unchanged; all Phase 3 fields were appended with
defaults, so every existing keyword-constructed fixture (test files,
`RegistryClient`) still builds correctly and now simply reports
`hardware_schema_version: 1` (accurately -- that's what a
Phase-1/2-shaped object is). The `Registry` interface's `Heartbeat`
method was kept exactly as it was; a new `HeartbeatWithMetrics` was
added alongside it rather than changing its signature. One existing
test, `TestRegister_SameAddressReregistrationStillAccepted`
(Phase 2), was not touched. See docs/HARDWARE_PROFILING.md for the
full design and docs/HARDWARE_CAPABILITIES.md for the field reference.

Built, all real and verified:

- **CPU profiling**: manufacturer + marketing model name (via the
  Windows registry, not just `platform.processor()`'s raw CPUID
  string), physical core count, logical processor count, live
  utilization %, architecture. Verified NOT to conflate physical and
  logical counts on Computer 1's hybrid-core CPU (10 physical / 12
  logical) -- a dedicated ground-truth test checks this exact pair of
  numbers on this specific machine.
- **RAM profiling**: total, available, used, utilization % -- all via
  `psutil.virtual_memory()`, refreshable live (see below).
- **GPU profiling**: multi-vendor detection (NVIDIA via `nvidia-smi`,
  authoritative for VRAM/driver; any vendor via PowerShell's
  `Get-CimInstance Win32_VideoController`, since `wmic` is deprecated/
  absent on this Windows build). Vendor identified from PCI vendor ID.
  Multiple GPUs supported (a list, not a single slot). Dedicated vs.
  shared memory distinguished (Intel integrated = `"shared"`). GPU
  presence is never conflated with CUDA support -- an Intel iGPU is
  detected and reported with an empty `backends` list.
- **VRAM profiling**: `memory_total_bytes` is `None`/omitted whenever
  genuinely unknown (WMI's `AdapterRAM` is `0`, missing, or not a
  plausible positive number) -- never a fabricated figure.
- **Inference backend capabilities**: `torch_installed` (a passive
  `import torch` check, nothing ever installed), and -- only if torch
  is present -- an actual `torch.cuda.is_available()` call and a
  `torch_directml` import check. All three are tri-state (`None` =
  untested, distinct from a tested `False`). Real, non-synthetic
  result on Computer 1: PyTorch (`2.10.0+cu126`) turned out to already
  be installed as a dependency of unrelated pre-existing software
  (`openai-whisper`, `torchvision` -- confirmed via `pip show torch`,
  not installed by this project), and `cuda_available` correctly
  tested `False` despite the CUDA-enabled build, because there's no
  NVIDIA GPU on this machine.
- **System information**: hostname (unchanged from Phase 1/2), OS
  name/version/build, architecture. Worker ID deliberately stays part
  of the registration envelope, not duplicated into the hardware
  payload.
- **Orchestrator integration**: `HardwareInfo` and `HeartbeatRequest`
  protos extended additively (new field numbers only, `optional` for
  anything tri-state), `hardware_schema_version` added,
  `WorkerRecord` extended with all Phase 3 fields,
  `GET /api/workers` now returns them (with Go pointer types +
  `omitempty` so "known false/zero" is distinguishable from "key
  absent, meaning unknown"), `GET /health` unchanged from Phase 2.
  Refreshing a snapshot reuses `Register()` (Phase 2 already made
  same-address re-registration safe) rather than adding a new RPC.
- **Live monitoring**: cheap CPU/RAM metrics refresh every heartbeat
  (`collect_live_metrics()`, psutil only); the expensive full snapshot
  (GPU enumeration) refreshes only every `hardware_refresh_interval_seconds`
  (default 60s) via a periodic re-`Register()` call. A dedicated test
  (`test_collect_live_metrics_never_calls_gpu_detection`) asserts the
  cheap path never touches GPU detection code at all.

Known limitations, disclosed:

- **AMD GPUs are untested** -- there is no AMD GPU anywhere in this
  project to verify detection against. AMD cards are identified via
  PCI vendor ID but tagged `memory_type: "unknown"` rather than
  `"dedicated"`, specifically because that can't be verified here.
- **WMI's `AdapterRAM` is a documented-unreliable 32-bit field**
  (misreports VRAM above ~4 GB, reports a static aperture rather than
  real allocation for integrated GPUs). NVIDIA is cross-checked
  against `nvidia-smi` to work around this; AMD has no equivalent
  cross-check available.
- **`directml_available` is a presence check only** -- it does not
  exercise the backend to confirm it actually works.
- Full-snapshot collection (`collect()`) measured at several seconds
  per call on Computer 1, dominated by `powershell.exe` process
  startup overhead, not the query itself -- this is exactly why it's
  rate-limited to once per `hardware_refresh_interval_seconds` rather
  than called per-heartbeat.

**Local verification vs. unverified remote hardware — reported
separately, as instructed:**
- **Computer 1: fully verified, live, via the real REST API.** See
  "Live REST API output" below.
- **Computer 2 / Computer 3: NOT verified.** Nothing about their CPU
  model, physical/logical core split, GPU vendor/model, or PyTorch/
  CUDA capability is known. docs/HARDWARE_PROFILING.md gives the exact
  commands to run the profiler there; nothing from Computer 1's result
  should be assumed to carry over.

### Phase 3 — second pass (hardware schema 3), 2026-10-31

Done against the full Phase 3 master prompt. It builds on the first
pass and preserves all Phase 1/2 behaviour. Every proto change is
additive, the `Registry` interface is unchanged, and `NewServer` /
`NewRegistryServer` keep their signatures (new `...WithOptions` /
`...WithCapability` variants were added alongside them). Details:
[HARDWARE_PROFILING.md](HARDWARE_PROFILING.md),
[HARDWARE_CAPABILITIES.md](HARDWARE_CAPABILITIES.md),
[SIMULATED_HARDWARE.md](SIMULATED_HARDWARE.md).

Built, real and verified:
- **Profile source and versioning**: every profile is `REAL`,
  `SIMULATED` or `UNKNOWN` (pre-schema-3 senders, never assumed real).
  Each carries a collection timestamp and `hardware_schema_version: 3`.
  Validation is versioned, and newer schemas are tolerated.
- **CPU**: frequency (Windows caveat below) and 1-minute load average
  (hidden during psutil's Windows warm-up).
- **Memory pressure**: page-file total/used/percent, refreshed on
  heartbeats.
- **GPU**: DXGI, called in-process via ctypes, added as a third
  source. It gives 64-bit memory figures, an explicit dedicated/shared
  split decided by measured size rather than name, and a stable GPU id
  (LUID). A WMI failure no longer empties the GPU list (detection
  failure ≠ absence). Iris Xe no longer reports its 2 GiB WMI aperture
  as `memory_total_bytes`, which corrects the first pass.
- **Live GPU metrics**: utilization and dedicated/shared memory in use,
  read from in-process PDH counters (~4 ms per sample, no subprocess),
  on their own 15 s interval.
- **Backend capability**: per-backend probe (cpu/cuda/xpu/mps/directml)
  with a verified 256×256 matmul test and GFLOPS. It runs **in a child
  process, once, in the background after the worker has registered**.
  While it runs, profiles are marked `backend_probe_pending`.
- **Simulated profiles**: YAML format with strict validation, three
  examples (A CPU-only, B GPU, C high-memory), `simulated_profile`
  worker setting, and a validate/show CLI. The orchestrator rejects
  them unless `allow_simulated_workers: true`, and they are labelled
  and counted separately everywhere.
- **Orchestrator**: hardware validation that rejects contradictions
  (listing every problem) but accepts incomplete data;
  `HeartbeatResponse.reregister_required`; per-GPU metric merging; new
  timestamps; capability classification (`internal/capability`,
  thresholds in config); partition-planning resources (available
  minus safety margin, VRAM only on compute-usable GPUs); new REST
  endpoints `/api/workers/{id}`, `/hardware`, `/metrics`,
  `/resources`, `/api/resources`, `?source=` filter; per-source counts
  in `/health`.
- **Reconnection**: a worker re-registers with a fresh profile after a
  network outage, or when the orchestrator restarted and lost its
  registry. Before this, a restarted orchestrator never saw running
  workers again; this was a real bug, found by reading Phase 2's code.
- **LAN readiness**: `advertise_host` setting (the worker's LAN IP when
  hostnames don't resolve between machines).

Problems found and fixed during this pass, disclosed:
- **Startup regression, caught by Phase 2's three-worker test.** With
  the backend probe (an `import torch`: 8.3 s cold, 13.9 s with three
  at once) on the registration path, three workers took 23–26 s to
  register, beyond the test's 15 s. Fixed by registering first and
  probing in a background child process. They now register in
  4.2–4.6 s, and worker RSS fell from 531 MiB to 53 MiB. The test was
  not changed.
- **Periodic refresh re-ran the probe** every 60 s. The probe now runs
  once per process, and the default refresh interval went from 60 s to
  300 s.
- **Orphaned probe children** when a worker is killed. Graceful
  shutdown now kills the child; a hard kill still can't (limitation
  below).
- **Intel Arc misclassification bug** in the new code, caught by a new
  test.
- **Wrong claim corrected**: I first documented Windows CPU frequency
  as "always the nominal 1700 MHz". A later capture showed 1457 MHz.
  It does drop when throttled; it just never shows turbo.

Existing tests changed, with reasons (no test removed, no assertion
weakened):
- `test_cpu_only_system_detect_gpus_returns_empty` and
  `test_incomplete_hardware_information_collect_still_succeeds`
  (`test_hardware_capabilities.py`): each now also stubs the new DXGI
  source. Without that stub they saw Computer 1's real Iris Xe and no
  longer simulated a GPU-less machine; assertions unchanged.
- `scripts/smoke_test_phase1.ps1`: a fixed 3 s sleep plus one check
  became a 15 s poll. The worker now registers at ~3.1 s (WMI GPU
  query on the startup path), which raced the old fixed wait. Pass
  criteria unchanged.
- `internal/config/config_test.go`: tests appended; existing ones
  untouched.

**Local vs. physical, reported separately as instructed:**
- **Local, Computer 1 only: done.** Unit tests, plus real-process
  integration tests over loopback (real orchestrator binary, real and
  simulated worker processes, orchestrator restart).
- **Physical multi-computer: NOT done.** Computers 2 and 3 have not
  participated. Their hardware is unknown, and no simulated profile
  describes them.

### Phase 4 — Hardware-aware model partitioning engine, 2026-10-31

A new, separate component: `model-partitioner/` (Python package
`partitioner`). It has no coupling to REST, gRPC, worker processes or
the frontend; it reads Phase 3 data and produces plan documents. **No
Phase 1–3 file was changed**, apart from one note added to
ARCHITECTURE.md §5.
Docs: [MODEL_PARTITIONING.md](MODEL_PARTITIONING.md),
[MEMORY_ESTIMATION.md](MEMORY_ESTIMATION.md),
[PARTITION_PLAN_SCHEMA.md](PARTITION_PLAN_SCHEMA.md).

Built, real and verified:
- **Inspection of the real GPT-2.** Its `config.json` (a few hundred
  bytes) is downloaded to the HF cache, and the real `GPT2LMHeadModel`
  is built on the meta device. **No weights were downloaded or loaded**;
  building the meta model cost +2 to +19 MiB of RSS. Results:
  - 12 layers, h=768, 12 heads, vocab 50,257, 1,024 positions.
  - **124,439,808 parameters**, matching an independent hand formula.
  - Tied `lm_head` ↔ `wte`, detected by tensor identity.
  - Cross-checked against the real checkpoint's safetensors **header**
    (a ranged read): 160 F32 tensors, 548,090,880 B, every parameter
    present with its shape. The 12 extra tensors are legacy causal-mask
    buffers, not parameters.
- **Architecture adapter interface**, with a strict GPT-2 adapter (it
  rejects unmapped modules, shape mismatches and block-count mismatches).
  Unsupported architectures are refused, not guessed.
- **Memory estimation.**
  - Covers weights (fp32/fp16/bf16 exact; int8/int4 estimated), tied-weight
    copies, KV cache, activation peaks, transfer buffers, load buffers, and
    a runtime overhead **measured** on Computer 1 (~700 MiB with torch +
    transformers imported).
  - Every item is tagged MEASURED / CALCULATED / ESTIMATED / UNKNOWN.
  - Plans state `quantization_runtime_implemented: false`.
- **Worker resources.** Taken from the live orchestrator, simulated YAML,
  or this machine. They use available RAM after margins, never total
  RAM. Planning is conservative on missing data; shared-host RAM is split;
  and a GPU is used only with known VRAM and known throughput.
  Python/Go parity is verified against the live orchestrator.
- **Optimizer.** Exact DP that minimizes the bottleneck stage time, then
  total time at that bottleneck. It searches every worker order for up
  to 6 workers and is deterministic. It matches a brute-force oracle on
  1,500 random instances.
- **Versioned plan schema v1 and an independent validator** that checks
  coverage, contiguity, order, duplicates, workers, source labels,
  memory (including recomputation) and the model fingerprint.
  Explanations include boundary sensitivity.
- **CLI** (`python -m partitioner inspect | memory | plan | validate`)
  and three example plans in `model-partitioner/examples/`.

**Real vs. simulated results, reported separately:**
- **Computer 1, REAL, measured live: GPT-2 is INFEASIBLE on it alone
  right now.**
  - Need: 1.56 GiB estimated for fp32 at context 1024.
  - Have: 1.45–1.47 GiB available during the runs (about 84% of RAM in
    use by the desktop), which leaves 0 B usable after the Phase 3
    margin.
  - The planner's own ~700 MiB was resident during the measurement, and
    the plan warns about this. Even at the 2.5 GiB available measured
    without it, effective capacity is about 0.84 GiB, still too little.
- **Simulated (A, B, C):** feasible plans with unequal, cost-balanced
  layer counts. These describe hypothetical hardware only, not
  Computers 2 and 3.

Mistakes found and fixed during the phase, disclosed:
- **Wrong test assumptions.** Two of my tests assumed a fast low-memory
  worker could help with two workers, and that a slow worker would go
  unused. The planner was right in both cases: end stages carry the
  embeddings / LM head, and even a slow worker lowers the bottleneck by
  taking `final_norm`. The tests were corrected to state those
  properties explicitly.
- **Misleading explanations.** Stage times that included link transfer
  were attributed entirely to compute. Compute and transfer are now
  split, and a test covers it.
- **Misleading unused reason.** Computer 1 was called "eligible but not
  needed" when it could hold no stage at all. The reason now says so.

### Phase 5 — Minimal inference execution, 2026-11-01

A new subpackage, `model-partitioner/partitioner/execution/`. It takes a
validated Phase 4 plan and runs **real inference in one process**.
Guide: [EXECUTION.md](EXECUTION.md).

- No Phase 1–3 file was changed: the worker, orchestrator, protos,
  registration and heartbeats are all untouched.
- Phase 4 changes are small:
  - `api.py` now sets `executable` from a rule instead of always false;
  - the CLI's estimator rebuild moved into
    `validate.estimator_for_plan` and is shared with the loader;
  - the CLI gained `execute`, `--model builtin:tiny-gpt2-v1`, and
    `--ram-margin-percent` / `--ram-margin-min-gib` for `--local`.
- No dependency was added or installed.

**Implemented and tested locally:**
- **Executable test model `builtin:tiny-gpt2-v1`.**
  - A GPT-2 config at toy size: 4 layers, hidden 32, vocab 128, 32
    positions, 56,000 parameters. The unchanged Phase 4 `GPT2Adapter`
    plans it.
  - Weights are a fixed formula (`formula-sin-v1`, no RNG, no download)
    and untrained.
  - The math is written out explicitly in PyTorch.
  - Verified against:
    - an independent float64 NumPy implementation (max |diff| 1.2e-4);
    - `transformers.GPT2LMHeadModel` loaded with the same weights
      (2.0e-4);
    - golden logits and next tokens `[60, 96]` (tolerance 5e-4, fp32 rounding).
- **Partitioned = unpartitioned.** Every 1-, 2-, 3- and 7-stage split
  matches `reference_forward` within the documented `REFERENCE_ATOL`
  of 1e-6. The observed difference is 0.0 (bit-identical). Each run
  repeats this check automatically.
- **Plan loading for execution.** It reuses `validate_plan`, with memory
  recomputed, and rejects (it never repairs):
  - malformed JSON, unsupported schema versions, and plans edited after
    planning (fingerprint);
  - infeasible plans, models without an implementation (GPT-2 included),
    and config or adapter mismatches;
  - quantized or non-fp32 plans, invalid ranges and missing workers;
  - in real mode: SIMULATED workers, stages assigned to another worker
    or host, and multi-stage plans.

  Plan provenance and warnings are carried into every result.
- **Two modes, labelled.**
  - `real` (`--local`): single stage, this machine's REAL worker.
  - `local-emulation` (`--emulate-stages-locally`): every stage in one
    process; worker assignments are not honored, and warnings say so.

  `distributed` is false in every result.
- **Stage interface v1** (`StageSpec`, `Activation`/`ActivationMeta`,
  `StageExecutor`, `ActivationTransport`). Shape, dtype, kind, producer,
  request id and finiteness are checked at every boundary by both the
  stage and the runner. The only transport is `InProcessTransport`
  (same object, no serialization).
- **Limits checked before allocation:** batch size, sequence length, the
  plan's own workload, the largest boundary tensor and per-unit
  activation peak (Phase 4 estimator), and weight bytes. Tests replace
  the weight and tensor constructors with functions that fail if called.
- **Structured results.** Status, mode, model, plan id, worker
  assignments, per-stage shapes and timings, output shape, next tokens,
  the reference check, and an error with phase, stage, every problem,
  the cause chain and the traceback.

**Real run on Computer 1 (2026-11-01):**
- **Default Phase 3 margins: INFEASIBLE.** The tiny model needs 700.32
  MiB, almost all of it the measured ~700 MiB torch + transformers
  runtime overhead. Computer 1 had 338.94 MiB effective capacity: 1.94
  GiB available, 376.60 MiB usable after the 10% / 1 GiB margin.
- **Margin deliberately reduced to 2% / 0.25 GiB** (the CLI prints a note
  about it). The plan was feasible, with 700.40 MiB needed of 1.44 GiB
  effective, and `execute --local computer-1` **SUCCEEDED**:
  - output `[2, 8, 128]` fp32, next tokens `[60, 96]`;
  - reference max |diff| 0.0;
  - stage wall time 1.8 ms;
  - total 11.3 s, mostly building the model spec (importing
    transformers). This is one run, not a benchmark.

**Not implemented:** remote tensor transport (no gRPC activation
messages, no tensor serialization), physical multi-machine inference,
GPT-2 distributed (or any) execution, quantized inference, KV cache and
generation, padding and attention masks.

### Phase 6 — Remote stage execution over gRPC, 2026-11-01

The first genuine remote execution path: a tiny-model request runs its
stages **in separate worker processes**, and activations travel between
them as gRPC messages. Guide: [REMOTE_EXECUTION.md](REMOTE_EXECUTION.md).

**Verification scope:**
- **Local processes on Computer 1 only** (127.0.0.1).
- **No physical multi-machine test was run.** Computers 2 and 3 are
  still unavailable.
- Results report the workers' self-reported hostnames and pids. Every
  run here shows one hostname, and the result text says it was not a
  multi-machine run.

**Implemented and tested locally:**
- **Protocol** `decentralizedllm.execution.v1.StageExecution`
  (`shared/proto/stage_execution.proto`, `PROTOCOL_VERSION_1`):
  - a single `ExecuteStage` RPC, separate from WorkerRegistry and
    Health, on its own server and port;
  - requests carry request id, plan id, the plan itself, model id,
    config fingerprint, target worker, stage index and unit range, and
    the input tensor;
  - responses carry a status (OK / REJECTED / BUSY / FAILED /
    DEADLINE_EXCEEDED), the output tensor, timing, a structured error
    with an `executed` flag, and the worker's self-reported host and pid.

  Python stubs and Go stubs (`orchestrator/internal/proto/execution`,
  unused for now) were generated **for this file only**, so no existing
  generated code changed. `scripts/generate_protos.ps1` now lists it.
- **Tensor format `raw-le-c-v1`** (`partitioner/execution/wire.py`):
  - little-endian, row-major, no header; fp32 and int64 only;
  - encoding, dtype, kind, rank, dimensions, byte length and size limit
    are checked before allocating;
  - finite values are checked right after;
  - no pickle.
- **Worker service** (`worker/worker/execution/`). Its checks, in order:
  - concurrency slot (`BUSY` if none), protocol version, fields, target
    worker id, required deadline;
  - plan hash and plan validation (cached per plan id);
  - **assignment**: the stage must be this worker id on this host,
    REAL, with the requested range;
  - tensor-vs-stage checks and input/output/batch/sequence limits, all
    before decoding;
  - execution of only that stage's components;
  - a deadline re-check after execution.

  Logs carry no tensor values, payload bytes or plan bodies. The bind
  policy allows loopback and private IPs, needs an opt-in for 0.0.0.0,
  and refuses public IPs and hostnames. A SIMULATED worker refuses to
  start the service. Two ways to run it:
  - `python -m worker.main` starts it only if `execution_enabled: true`;
  - `python -m worker.execution` runs it alone.
- **Coordinator** (`partitioner/execution/remote.py`,
  `execute --remote --endpoint ID=HOST:PORT`):
  - validates the plan (Phase 5 loader, new `remote` mode) and requires
    an endpoint per stage worker;
  - checks the input and limits, then sends the stages in order;
  - checks every response: version, request id, the **responding worker
    id**, stage index, and the output tensor;
  - compares the final logits with the Phase 5 local reference
    (`REMOTE_REFERENCE_ATOL = 1e-4`; observed **0.0**);
  - **retries only requests known not to have executed** (channel never
    ready, BUSY, gRPC RESOURCE_EXHAUSTED), up to 3 attempts by default;
  - reports deadline, mid-call loss and worker FAILED as
    `AMBIGUOUS_FAILURE` or the worker's error code, **not retried**.

**Observed on Computer 1** (one run, not a benchmark):
- two `python -m worker.execution` processes ran stage 0 (units [0, 3))
  and stage 1 (units [3, 7));
- output `[2, 8, 128]`, next tokens `[60, 96]`, reference diff 0.0;
- requests about 12.5 / 14.5 KB, mostly the plan JSON; responses 2.2 /
  8.3 KB;
- round trips 128 ms (cold caches) and 26 ms;
- each service-only process took about 11 s to start (torch +
  transformers import).

**Changes to earlier phases:**
- **Worker:** `config/loader.py` gained `execution_*` settings, off and
  loopback by default. `main.py` gained a few lines to start and stop
  the optional server. **Registration, heartbeats and Health are
  unchanged**: the registered address is still the Health server, as the
  integration test asserts.
- **Partitioner:**
  - the loader gained `REMOTE_MODE`;
  - the Phase 5 runner refuses remote mode explicitly;
  - two messages now point to `--remote`: the multi-stage reason in the
    planner's `executable_reason`, and the local-mode loader problem.
    Existing assertions still hold.
- **CLI:** `--remote`, `--endpoint`, `--stage-timeout`,
  `--connect-timeout`, `--max-attempts`. Local modes are unchanged.
- **Config and docs:** `worker.example.yaml`, `.env.example`,
  `NETWORKING.md`, `EXECUTION.md` and the partitioner README were
  updated.
- Nothing was installed, and no model was downloaded.

**Not implemented:**
- physical multi-machine validation;
- authentication or TLS (the protocol is **unauthenticated and
  unencrypted**);
- endpoint discovery through the orchestrator;
- automatic placement, fault-tolerant rescheduling or KV cache;
- GPT-2 execution, quantized inference.

### Phase 7 — Physical LAN validation readiness, 2026-11-01

Goal: get the two-computer pipeline ready, validate everything possible
on Computer 1, and write a repeatable procedure for Computer 2. Guide:
[DEPLOYMENT_LAN.md](DEPLOYMENT_LAN.md).

**The physical multi-machine test is `PENDING`.** Computers 2 and 3 are
unavailable, and no test or run is `REAL_MULTI_MACHINE`.

**Implemented and tested:**
- **Loopback by default everywhere.** The orchestrator REST/gRPC
  (`0.0.0.0:8080/9090` → `127.0.0.1`) and worker Health (`0.0.0.0` →
  `127.0.0.1`) defaults changed; StageExecution already was loopback.
  One bind policy for every listener:
  - loopback and private IPs are allowed;
  - public IPs and hostnames are refused at startup;
  - `0.0.0.0` is allowed with a warning (orchestrator, Health) or needs
    an explicit opt-in (StageExecution).

  It is implemented in Go (`config.CheckBindAddress`) and Python
  (`worker/config/bind_policy.py`, shared by Health and StageExecution).
  The example configs and `.env.example` now use loopback too.
- **`GetInfo` RPC** on StageExecution: read-only; reports protocol
  versions, identity, models, encodings, limits and
  `authenticated: false`. Python and Go stubs were regenerated for that
  proto only. Phase 6 workers are handled by a rejected-version probe
  that executes nothing.
- **`python -m partitioner preflight`** (`partitioner/preflight.py`).
  Checks:
  - runtime and pins;
  - this machine's interfaces, with a public-IP warning;
  - worker config: parse, stable id, bind policy, address ownership,
    port free, port clash, execution and simulated flags, message limit,
    orchestrator address;
  - orchestrator: DNS and TCP, `/health`, registrations online and REAL,
    backend probe finished;
  - endpoints: DNS and TCP with classified failures (dns, refused,
    timeout, unreachable, Windows loopback quirk), gRPC GetInfo, protocol
    compatibility, identity;
  - plan: validity, capacity against the live orchestrator, per-stage
    assignment against GetInfo and the orchestrator, loopback-for-remote
    detection, hosts spanned (`--require-multi-host`);
  - limits per stage against worker GetInfo and coordinator defaults.

  Every check states what it tested; anything untested is `SKIP`. Exit 1
  on any FAIL.
- **Test classification.** Every Python test is classified, enforced at
  collection, with a summary at the end of each run
  (`tests/classification.py`). Remote results carry `test_classification`
  derived from worker self-reports; local results carry
  `REAL_SINGLE_MACHINE` or `SIMULATED`.
- **`scripts/phase7_local_validation.py`.** With the real orchestrator
  and 3 `worker.main` processes on loopback, it runs:
  - preflight, exposure audit, planning from live registrations;
  - 2- and 3-worker pipelines;
  - limit, malformed and timeout failures;
  - graceful and hard worker restarts, orchestrator restart;
  - clean shutdown.

  It writes `report.md`, `report.json` and the logs.
  `scripts/lan_result_summary.py` turns an execution result into the
  physical-test record rows.

**Local validation results** (`REAL_MULTI_PROCESS`, Computer 1):
[run2](validation/phase7-local-2026-11-01-run2/report.md) passed
**14/14**. Run 1 passed 13/14 because of a script bug (wrong stage-owner
assumption), which was fixed and is kept on record. Highlights:
- all 8 listening sockets were on 127.0.0.1;
- pipelines matched the reference exactly (diff 0.0);
- a hard-killed worker was marked offline after 2.8 s and back 0.5 s
  after restart;
- a graceful restart took 8.1 s; while the worker was down, its stage
  failed `RETRIES_EXHAUSTED`, `executed: false`;
- **execution kept working while the orchestrator was down**, and all
  workers re-registered 0.8 s after it returned.

**Findings, not fixed:**
- **Live planning on Computer 1 is INFEASIBLE.** The largest effective
  capacity was 52–82 MiB across runs, against 700 MiB per stage (the
  measured runtime overhead). The validation pipelines used the same REAL
  records with a synthetic 3 GiB capacity, labelled as such everywhere.
- **Firewall exposure.** Computer 1 has two pre-existing inbound
  **Allow** rules for `python.exe` on the **Public** firewall profile
  ("Python"). They were not changed, and the guide explains how to
  review them.
- Workers register before their backend probe finishes, and planning
  refuses them until it does (about 4 s here; preflight now warns). The
  first validation dry run hit this.
- On Windows a closed port reports "refused" only after about 2 s
  (measured), so preflight's default timeout is 3 s.

**Not done:** the physical two-computer test (procedure and `PENDING`
results table in DEPLOYMENT_LAN.md); the optional LAN-IP bind step
(could raise a firewall prompt, and its traffic stays on the machine);
TLS and authentication (future work, described, not implemented).

### Phase 8 — Two-machine LAN test readiness, 2026-11-01

Guide: [DEPLOYMENT_LAN.md](DEPLOYMENT_LAN.md). Report:
[validation/PHASE8_REPORT.md](validation/PHASE8_REPORT.md). **No second
computer took part, so the physical test is still `PENDING`, and nothing
is `REAL_MULTI_MACHINE`.**

**Implemented and tested:**
- **Orchestrator (Go).** It records each registration's TCP peer IP as
  `observed_peer_ip`, uses it in duplicate-id detection, and exposes it
  in `GET /api/workers`.

  This fixes a real hazard. With Health on loopback, two machines both
  report `127.0.0.1:50100`, so a duplicated `worker_id` would have
  silently replaced the other machine's registration.
- **Preflight additions:**
  - `--orchestrator-config` (bind policy mirroring Go, address
    ownership, port, simulated workers);
  - a local Windows Firewall check via `netsh` (0.4 s vs. 65 s for the
    cmdlets): missing rule, Block override, broad rule, Public-profile
    program rules; SKIP when the output can't be parsed;
  - the port owner via psutil (FAIL when an unrelated program holds it);
  - `--probe-orchestrator` from the worker machine;
  - peer-IP notes in the plan check;
  - `--peer-ip`, which fills in exact firewall commands in hints.
- **`scripts/phase8_lan.py`:**
  - config generators that refuse public, `0.0.0.0`, hostname, port
    clashes and overwrites;
  - `run`, the two-machine test: wait for readiness → preflight → plan
    from live registrations only → preflight with
    `--require-multi-host` → N executions against the reference →
    evidence-based classification → guided disconnect/restart;
  - `rehearse`, a loopback version that is always REAL_MULTI_PROCESS.

  `REAL_MULTI_MACHINE` requires 2+ distinct reported hostnames **and** 2+
  distinct non-loopback registration peer IPs.
- **LAN guide rewritten:**
  - finding the IPs, generating configs, least-privilege firewall rules;
  - preflight with expected output, startup order with the log line to
    wait for, the run, shutdown;
  - a 22-row troubleshooting table and the checklist with its `PENDING`
    results table.

**Local results:**
- loopback rehearsal: P1–P8 PASS, classification **REAL_MULTI_PROCESS**;
- strict rehearsal: correctly FAILS the multi-machine criterion;
- Phase 7 validation re-run: 14/14.

**Findings on Computer 1, not modified:**
- Public-profile Allow rules exist for `python.exe` and
  `orchestrator.exe` on any port;
- port 8080 is held by an unrelated `java.exe`;
- Windows excludes TCP ports 50000–50059;
- live planning is infeasible under the default margins.

## In progress

None. Phase 8 is complete apart from the physical two-computer test,
which is `PENDING` until Computer 2 is available (manual steps:
PHASE8_REPORT.md). Nothing was committed or pushed, and nothing was
installed.

## Test results (actual output, Phase 8, 2026-11-01)

No physical multi-machine test was run.

| Suite | Command | Result | REAL_SINGLE_MACHINE | REAL_MULTI_PROCESS | REAL_MULTI_MACHINE | SIMULATED |
|---|---|---|---|---|---|---|
| Partitioner | `cd model-partitioner; python -m pytest -q` | **383 passed**, 0 failed | 233 | 14 | 0 | 136 |
| Worker | `cd worker; python -m pytest -q` | **150 passed**, 0 failed | 57 | 3 | 0 | 90 |
| Go | `cd orchestrator; gofmt -l .; go vet ./...; go test ./... -v -count=1` | clean, **111 `--- PASS`, 0 `--- FAIL`** | in-process | | 0 | |
| Phase 1 smoke | `scripts/smoke_test_phase1.ps1` | **PASS** | | yes | | |
| Phase 7 validation (regression) | `python scripts/phase7_local_validation.py --out <scratch>` | **14/14** | | yes | | |
| Phase 8 rehearsal | `python scripts/phase8_lan.py rehearse --out docs/validation/phase8-loopback-rehearsal-2026-11-01 --repeat 3` | **PASS** | | yes | 0 | |
| Phase 8 strict rehearsal | `… rehearse --strict --no-disruption --repeat 1 --out docs/validation/phase8-loopback-strict-2026-11-01` | **FAIL by design** (criterion not met) | | yes | 0 | |

## Test results (actual output, Phase 7, 2026-11-01)

No physical multi-machine test was run.

| Suite | Command | Result | REAL_SINGLE_MACHINE | REAL_MULTI_PROCESS | REAL_MULTI_MACHINE | SIMULATED |
|---|---|---|---|---|---|---|
| Partitioner | `cd model-partitioner; python -m pytest -q` | **326 passed**, 0 failed | 209 | 7 | 0 | 110 |
| Worker | `cd worker; python -m pytest -q` | **150 passed**, 0 failed | 57 | 3 | 0 | 90 |
| Go | `cd orchestrator; gofmt -l .; go vet ./...; go test ./... -v -count=1` | gofmt and vet clean, **104 `--- PASS`, 0 `--- FAIL`** | in-process tests, not individually classified (none multi-process or multi-machine) | | 0 | |
| Phase 1 smoke | `powershell -ExecutionPolicy Bypass -File scripts/smoke_test_phase1.ps1` | **PASS** | | real orchestrator + worker | | |
| Local validation | `python scripts/phase7_local_validation.py --out docs/validation/phase7-local-2026-11-01-run2` | **14/14** | 1 step | 13 steps | 0 | |

New tests:
- `test_preflight.py` (35);
- `worker/tests/test_bind_policy.py` (13);
- 3 Go config tests (loopback defaults, bind policy, public address
  refused);
- `test_classification` assertions added to existing execution tests.

During development, one new test failed intermittently: the live
preflight test raced the workers' backend probes. The test was wrong
(it waited for "online" but asserted "probe finished") and now waits for
the right state. It passed in 3 consecutive runs afterwards.

## Test results (actual output, Phase 6, 2026-11-01)

All on Computer 1. **Local-process tests only; no physical multi-machine test was run.**

**Partitioner** (`cd model-partitioner; python -m pytest -q`): **291 passed, 0 failed, 0 skipped**
```
tests/test_remote_wire.py         24  (new: round trips, exact byte layout, malformed payloads, size limit before allocation, NaN/inf, no pickle)
tests/test_remote_service.py      38  (new: worker service over real gRPC in-process: version, malformed, wrong worker,
                                       unassigned stage, plan hash/validity, SIMULATED, stage mismatch, host, tensors,
                                       limits, deadline required/too short/exceeded, BUSY, FAILED, message size, logs)
tests/test_remote_execution.py    28  (new: Part A in-process servers: 2/3-stage pipelines, missing endpoint, misrouting,
                                       unreachable -> bounded retries, BUSY -> retry -> OK, ambiguous deadline not retried,
                                       failure not retried, reference mismatch, CLI;
                                       Part B SEPARATE PROCESSES: 2- and 3-worker pipelines over real gRPC, CLI,
                                       killed worker process)
tests/test_remote_worker_main.py   1  (new: real Go orchestrator + 2 `worker.main` processes with execution enabled:
                                       register, execute, keep heartbeating, graceful shutdown)
Phase 4-5 files                  200  (unchanged assertions)
```

**Worker** (`cd worker; python -m pytest -q`): **137 passed** (120 earlier + 17 new in `tests/test_execution_config.py`).

**Go** (`cd orchestrator; gofmt -l .; go vet ./...; go test ./... -v -count=1`): gofmt clean, vet clean, **101 `--- PASS`, 0 `--- FAIL`**.

**Phase 1 smoke test** (`powershell -ExecutionPolicy Bypass -File scripts/smoke_test_phase1.ps1`): **PASS**.

## Test results (actual output, Phase 5, 2026-11-01)

**Partitioner** (`cd model-partitioner; python -m pytest -q`): **200 passed, 0 failed, 0 skipped**
```
tests/test_cli.py               4
tests/test_execution_cli.py    11   (new)
tests/test_execution_model.py  36   (new: structure, weights, golden output, NumPy + HF oracles, all splits, stage/tensor validation)
tests/test_execution_plans.py  18   (new: valid/invalid loading, simulated refusal, executable flag)
tests/test_execution_runner.py 22   (new: real + emulated runs, limits before allocation, structured errors)
tests/test_inspection.py       18
tests/test_memory.py           25
tests/test_optimizer.py         9
tests/test_planning.py         31   (Phase 4 assertions unchanged)
tests/test_validate.py         19
tests/test_workers.py           7
```

**Regression** (no Phase 1–3 code changed):
- Worker (`cd worker; python -m pytest -q`): **120 passed**.
- Go (`cd orchestrator; go vet ./...; go test ./... -v -count=1`): vet
  clean, **101 `--- PASS`, 0 `--- FAIL`**.
- All three Phase 4 example plans still pass
  `python -m partitioner validate`, with memory recomputed.

## Test results (actual output, Phase 4, 2026-10-31)

**Partitioner** (`cd model-partitioner; python -m pytest`):
```
tests/test_cli.py          4
tests/test_inspection.py  18   (real GPT-2 + checkpoint header + synthetic/invalid configs)
tests/test_memory.py      25
tests/test_optimizer.py    9   (incl. 1,500-instance brute-force oracle, 200-instance secondary-objective check)
tests/test_planning.py    31   (incl. 40-cluster randomized property test)
tests/test_validate.py    19
tests/test_workers.py      7   (incl. live Go orchestrator parity, real Computer 1 profiling)
113 passed, 0 skipped
```

**Phase 1–3 regression** (re-run after Phase 4; no Phase 1–3 code changed):
- Go: `go vet` clean, `ok` for api, capability, config, grpcserver and
  registry; **101 `--- PASS`, 0 FAIL**.
- Worker Python suite: **120 passed**.
- Phase 1 smoke test: **PASS**.

All three example plans pass `python -m partitioner validate`, with
memory recomputed.

## Test results (actual output, Phase 3 second pass, 2026-10-31)

**Go** (`cd orchestrator; go build ./...; go vet ./...; gofmt -l .; go test ./... -v -count=1`):
```
build exit=0
vet exit=0
gofmt: []
ok  	orchestrator/internal/api
ok  	orchestrator/internal/capability
ok  	orchestrator/internal/config
ok  	orchestrator/internal/grpcserver
ok  	orchestrator/internal/registry
PASS: 101  FAIL: 0  top-level funcs: 84
```
84 top-level tests (up from 43), 101 `--- PASS` lines including
subtests. New: `registry_schema3_test.go`,
`capability/capability_test.go`, `registry_server_schema3_test.go`,
`rest_schema3_test.go`, plus 4 appended to `config_test.go`.
`go test -race` is still unavailable (no cgo toolchain; unchanged
since Phase 2).

**Python** (`cd worker; python -m pytest -v`, against the rebuilt `orchestrator.exe`):
```
tests/test_config_loader.py                      4
tests/test_config_phase3.py                      3    (new)
tests/test_graceful_shutdown_integration.py      1
tests/test_hardware_capabilities.py              15
tests/test_hardware_profiler.py                  2
tests/test_hardware_schema3.py                   52   (new)
tests/test_main_retry.py                         3
tests/test_phase3_hardware_integration.py        3    (new, real processes)
tests/test_registry_client.py                    3
tests/test_simulated_profiles.py                 33   (new)
tests/test_three_worker_integration.py           1
======================= 120 passed in 70.47s =======================
```
0 skipped: every Computer 1 ground-truth test actually ran. Up from 29.

**Phase 1 smoke test** (`scripts/smoke_test_phase1.ps1`): `PASS: worker
'smoke-test-worker' is registered with real hardware data
(cpu_logical_cores=12, hostname=YakobsPC, has_cuda_gpu=False)`, after
the polling change explained above.

**Frontend**: not touched this phase; not re-run.

**Measured overhead** (idle real worker, default intervals, 60 s
window after startup): 0.36 s CPU = **0.60% of one core, 53 MiB RSS**.
Component costs: live metrics 0.23 ms CPU, PDH GPU sample 4.2 ms, DXGI
enumeration 1 ms.

**Computer 1, live `GET /api/workers/computer-1`** (final build,
2026-10-31 07:15, loopback; `backend_capabilities` trimmed to the
key fields):
```json
{
 "id": "computer-1", "address": "127.0.0.1:19240", "hostname": "YakobsPC",
 "cpu_logical_cores": 12, "total_ram_bytes": 16876888064, "has_cuda_gpu": false,
 "gpu_name": "", "gpu_vram_bytes": 0, "os": "Windows 11", "python_version": "3.13.3",
 "status": "idle", "online": true, "last_heartbeat_age_seconds": 1.05,
 "registered_at": "2026-10-31T07:14:45.5716185+05:30", "last_heartbeat": "2026-10-31T07:15:21.1678891+05:30",
 "hardware_schema_version": 3, "cpu_manufacturer": "Intel",
 "cpu_model": "13th Gen Intel(R) Core(TM) i7-1355U", "cpu_physical_cores": 10, "cpu_architecture": "AMD64",
 "os_name": "Windows", "os_version": "11", "os_build": "10.0.26200",
 "gpus": [{
   "vendor": "Intel", "name": "Intel(R) Iris(R) Xe Graphics", "memory_type": "shared", "backends": [],
   "driver_version": "32.0.101.6881", "id": "luid:0x00000000_0x000160E5",
   "dedicated_memory_bytes": 134217728, "shared_memory_bytes": 8438444032, "memory_source": "dxgi",
   "integrated": true, "utilization_percent": 12.53, "dedicated_memory_used_bytes": 0,
   "shared_memory_used_bytes": 1464348672 }],
 "torch_installed": true, "torch_version": "2.10.0+cu126", "cpu_backend_available": true,
 "cuda_available": false, "cuda_device_count": 0, "directml_available": false,
 "cpu_utilization_percent": 14.8, "ram_available_bytes": 3636699136, "ram_used_bytes": 13240188928,
 "ram_utilization_percent": 78.5,
 "profile_source": "REAL", "profile_collected_at": "2026-10-31T07:14:58.924+05:30",
 "hardware_updated_at": "2026-10-31T07:14:58.9579885+05:30",
 "metrics_updated_at": "2026-10-31T07:15:21.1678891+05:30", "metrics_collected_at": "2026-10-31T07:15:21.167+05:30",
 "cpu_frequency_current_mhz": 1457, "cpu_frequency_max_mhz": 1700,
 "swap_total_bytes": 17398558720, "swap_used_bytes": 3047477247, "swap_utilization_percent": 17.5,
 "gpu_count": 1, "backend_probe_pending": false,
 "backend_capabilities": [
  {"name": "cpu", "available": true, "usable": true, "tested": true, "test_passed": true, "test_gflops": 31.05, "detail": "256x256 float32 matmul verified"},
  {"name": "cuda", "available": false, "usable": false, "version": "12.6", "tested": false, "detail": "torch built for CUDA 12.6, but torch.cuda.is_available() is False (no usable NVIDIA GPU/driver)"},
  {"name": "xpu", "available": false, "usable": false, "tested": false, "detail": "torch.xpu.is_available() is False (no Intel GPU runtime in this torch build)"},
  {"name": "mps", "available": false, "usable": false, "tested": false, "detail": "Apple Metal backend not available (not an Apple Silicon Mac)"},
  {"name": "directml", "available": false, "usable": false, "tested": false, "detail": "torch-directml package not installed"}],
 "torch_cuda_build_version": "12.6",
 "classification": {"classes": ["CPU_ONLY"], "memory_tier": "STANDARD", "accelerator_backends": [],
  "reasons": ["CPU_ONLY: CPU backend works and no accelerator backend is usable",
   "cuda unavailable: torch built for CUDA 12.6, but torch.cuda.is_available() is False (no usable NVIDIA GPU/driver)",
   "...xpu, mps, directml likewise...",
   "GPU \"Intel(R) Iris(R) Xe Graphics\" (Intel) is present but no usable compute backend targets it",
   "memory tier STANDARD: 15.72 GiB installed, between 8.00 and 32.00 GiB thresholds"]}
}
```
`GET /health` in the same run:
`{"status":"ok","registered_workers":2,"online_workers":2,"online_real_workers":1,"online_simulated_workers":1,"online_unknown_source_workers":0}`.
The simulated worker (`sim-profile-b`) appeared with
`profile_source: "SIMULATED"`, `profile_notice: "SIMULATED - NOT
PHYSICALLY VERIFIED"`, classes `GPU_ACCELERATED, HIGH_MEMORY`, and it
was listed under `"simulated"` in `/api/resources`, never under
`"real"`. More responses: HARDWARE_PROFILING.md, "REST API".

## Test results (actual output, Phase 3 first pass, 2026-10-30)

**Go (`cd orchestrator; go build ./... && go vet ./... && go test ./... -v`):**
```
build exit=0
vet exit=0
ok  	orchestrator/internal/api
ok  	orchestrator/internal/config
ok  	orchestrator/internal/grpcserver
ok  	orchestrator/internal/registry
```
43/43 passed (`--- PASS` count), up from 33 in Phase 2. New this
phase: `registry_hardware_test.go` (4 tests), `rest_hardware_test.go`
(5 tests), `registry_server_hardware_test.go` (3 tests). `gofmt -l .`
clean after `gofmt -w .`.

**Python (`cd worker; python -m pytest -v`):**
```
29 passed in 18.99s
```
Up from 14 in Phase 2. New this phase:
`tests/test_hardware_capabilities.py` (15 tests covering CPU-only
systems, missing NVIDIA drivers, nvidia-smi erroring/timing out,
PowerShell unavailable, malformed PowerShell output, incomplete
hardware information, PNP vendor-ID parsing, NVIDIA/WMI dedup, unknown
VRAM reported as `None` not `0`, torch-absent tri-state behavior, and
the live-metrics-never-touches-GPU-detection guarantee). Both Phase
2 real-process integration tests (`test_graceful_shutdown_integration.py`,
`test_three_worker_integration.py`) re-run and still pass against the
rebuilt `orchestrator.exe`, confirming the Register/Heartbeat proto
changes didn't break the wire contract.

**Hardware profiler run standalone on Computer 1 (real output):**
```json
{
  "hostname": "YakobsPC",
  "cpu_logical_cores": 12,
  "total_ram_bytes": 16876888064,
  "has_cuda_gpu": false,
  "os": "Windows 11",
  "python_version": "3.13.3",
  "hardware_schema_version": 2,
  "cpu_manufacturer": "Intel",
  "cpu_model": "13th Gen Intel(R) Core(TM) i7-1355U",
  "cpu_physical_cores": 10,
  "cpu_architecture": "AMD64",
  "gpus": [
    {
      "vendor": "Intel",
      "name": "Intel(R) Iris(R) Xe Graphics",
      "memory_total_bytes": 2147479552,
      "memory_type": "shared",
      "backends": [],
      "driver_version": "32.0.101.6881"
    }
  ],
  "torch_installed": true,
  "torch_version": "2.10.0+cu126",
  "cuda_available": false,
  "cuda_device_count": 0,
  "directml_available": false,
  "ram_utilization_percent": 82.7
}
```
(Full output, including all live-metric fields, is longer; trimmed
here for readability -- see docs/HARDWARE_PROFILING.md for the exact
command to reproduce.) `cpu_physical_cores` (10) is correctly distinct
from `cpu_logical_cores` (12), as required.

**Live REST API output (real orchestrator + real worker, Computer 1,
loopback)**, `GET /api/workers`:
```json
[{
  "id":"computer-1-demo","address":"127.0.0.1:19201","hostname":"YakobsPC",
  "cpu_logical_cores":12,"total_ram_bytes":16876888064,"has_cuda_gpu":false,
  "os":"Windows 11","python_version":"3.13.3","status":"idle","online":true,
  "last_heartbeat_age_seconds":0.27,
  "hardware_schema_version":2,"cpu_manufacturer":"Intel",
  "cpu_model":"13th Gen Intel(R) Core(TM) i7-1355U","cpu_physical_cores":10,
  "cpu_architecture":"AMD64","os_name":"Windows","os_version":"11","os_build":"10.0.26200",
  "gpus":[{"vendor":"Intel","name":"Intel(R) Iris(R) Xe Graphics",
    "memory_total_bytes":2147479552,"memory_type":"shared","backends":[],
    "driver_version":"32.0.101.6881"}],
  "torch_installed":true,"torch_version":"2.10.0+cu126","cpu_backend_available":true,
  "cuda_available":false,"cuda_device_count":0,"directml_available":false,
  "cpu_utilization_percent":6.8,"ram_available_bytes":3278569472,
  "ram_used_bytes":14148952064,"ram_utilization_percent":83.8
}]
```
Verified live-refresh: `cpu_utilization_percent` was `0` immediately
after the worker's first heartbeat (the documented "nothing to compare
against yet" behavior of `psutil.cpu_percent(interval=None)`'s first
call) and `6.8` after subsequent heartbeats a few seconds later --
confirming the live metrics genuinely update, not just appear once.
`GET /health` showed `"registered_workers":1,"online_workers":1`
throughout.

## Test results (actual output, Phase 2, 2026-10-30)

**Go (`cd orchestrator; go build ./... && go vet ./... && go test ./... -v`):**
```
build exit=0
vet exit=0
ok  	orchestrator/internal/api         (6 tests)
ok  	orchestrator/internal/config      (5 tests)
ok  	orchestrator/internal/grpcserver  (6 tests)
ok  	orchestrator/internal/registry    (16 tests)
```
33/33 passed (confirmed via `--- PASS` count, matching `grep -c "^func Test"`).
`go test ./... -race` was attempted and could not run: this machine
has no 64-bit C toolchain, which cgo (required by Go's race detector)
needs (`cc1.exe: sorry, unimplemented: 64-bit mode not compiled in`).
Not installed without being asked — see "Known limitation" above.

**Python (`cd worker; python -m pytest -v`):**
```
tests/test_config_loader.py ....                              4 passed
tests/test_graceful_shutdown_integration.py .                  1 passed
tests/test_hardware_profiler.py ..                              2 passed
tests/test_main_retry.py ...                                    3 passed
tests/test_registry_client.py ...                               3 passed
tests/test_three_worker_integration.py .                        1 passed
============================== 14 passed in 5.83s ==============================
```
All 14 passed, including both real-process integration tests, run
against a freshly rebuilt `orchestrator.exe`. No stray `orchestrator`
or `python -m worker.main` processes remained afterward (checked via
`Get-Process` / `Get-CimInstance Win32_Process`).

**Frontend (`cd frontend; npm run build; npm run lint`):**
Unchanged from Phase 1 — not touched this phase. Build succeeded,
static page generated, zero lint errors (verified in Phase 1; no
frontend files were modified in Phase 2, so not re-run).

**`scripts/multi_worker_test.py` (manual runner, smoke-checked):**
Run for ~10s: orchestrator started, all three workers (`manual-worker-1/2/3`)
registered successfully with real hardware data, logs confirmed
`connecting to orchestrator...` / `Register(...) -> accepted=True`
for each. The script's own Ctrl+C teardown path was not separately
exercised interactively in this session (it was started
non-interactively for this check) — it calls the identical
`CTRL_BREAK_EVENT` → deregister code path that
`test_graceful_shutdown_integration.py` already proved works, so this
is reported as "startup verified directly; teardown verified by
shared code path," not "fully verified standalone."

**Phase 1 local smoke test (`scripts/smoke_test_phase1.ps1`):**
```
Orchestrator /health: {"status":"ok","uptime_seconds":1.0154467,"registered_workers":0}
Orchestrator /api/workers: {"id":"smoke-test-worker","address":"127.0.0.1:19100",
  "hostname":"YakobsPC","cpu_logical_cores":12,"total_ram_bytes":16876888064,
  "has_cuda_gpu":false,"gpu_name":"","gpu_vram_bytes":0,"os":"Windows 11",
  "python_version":"3.13.3","status":"idle", ...}
PASS: worker 'smoke-test-worker' is registered with real hardware data
  (cpu_logical_cores=12, hostname=YakobsPC, has_cuda_gpu=False)
```
First run failed with a port-bind conflict caused by a path-quoting bug
in the test script (Start-Process did not quote a path containing a
space); fixed, re-run passed. The one known limitation: `Stop-Process`
force-kills rather than sending Ctrl+C, so the worker's graceful
deregister-on-shutdown handler does not get a chance to run under this
specific test-teardown method — this is a limitation of how the test
stops the process, not of the deregistration code itself (Ctrl+C in an
interactive terminal does trigger it correctly, confirmed separately
via `go test`/`pytest` unit coverage of the deregister path).

## Known issues / blockers

- (Phase 8) **The physical two-computer test is still PENDING.** All
  tooling is ready (DEPLOYMENT_LAN.md, Steps 1–8).
- (Phase 8) Heartbeats are still matched by worker id only. The
  observed peer IP is checked at registration, not on every heartbeat.
- (Phase 8) The firewall check parses English `netsh` output; on other
  languages it reports SKIP. It is a prediction, and the real test is a
  connection from the peer.
- (Phase 8) On Computer 1, port 8080 is held by an unrelated Java
  process, so use another `--rest-port` while it runs.

- (Phase 7) **The physical two-computer test is PENDING.** Computer 2 is
  unavailable. Procedure: DEPLOYMENT_LAN.md.
- (Phase 7) Live planning on Computer 1 is infeasible under the default
  Phase 3 margins: 52–82 MiB effective vs. 700 MiB per stage. A real C1+C2
  plan may need freed RAM or explicitly lowered orchestrator `capability:`
  margins, recorded as a deviation.
- (Phase 7) Computer 1 has inbound Allow rules for `python.exe` on the
  Public firewall profile. Review them; they were not modified.
- (Phase 7) Default binds changed from `0.0.0.0` to `127.0.0.1` for the
  orchestrator and worker Health. Any setup that relied on the old
  defaults to reach another machine must now set the LAN IP explicitly.

- (Phase 6) Remote execution has run **only between processes on
  Computer 1**. Cross-machine behaviour, firewall rules for port 50200
  and cross-CPU numerical agreement are untested. The 1e-4 tolerance is
  reasoned, not measured.
- (Phase 6) StageExecution is **unauthenticated and unencrypted**.
  Identity and assignment checks prevent misrouting, not attacks.
- (Phase 6) Endpoints are passed by hand. The orchestrator does not know
  them, because registration was deliberately left unchanged.
- (Phase 6) Each worker process costs about 700 MiB (torch +
  transformers). Several processes at once strain Computer 1, which runs
  at about 80% RAM use.

- (Phase 5) Execution exists only for the built-in tiny model, with
  untrained formula weights, fp32, a single prefill pass and no KV
  cache. Multi-stage execution is single-process emulation only.
- (Phase 5) Computer 1 can't host even the tiny model under the default
  Phase 3 margin, because every stage carries the measured ~700 MiB
  runtime overhead. The real run used a deliberately reduced margin. The
  overhead figure was measured with transformers imported; a stage-only
  process may need less, but that has not been measured.
- (Phase 5) Execution lives in `model-partitioner/`, not in the worker
  process. `worker/worker/execution` and `worker/worker/inference` are
  still empty.

- (Phase 4) The partitioner's cost model is compute-only and uses a
  noisy CPU micro-benchmark or a core-count assumption. GPU throughput
  must be supplied by hand. Seconds in plans are relative estimates, not
  predictions.
- (Phase 4) Only GPT-2 has an adapter. Quantized sizes are estimates,
  with no quantized runtime. Activation estimates assume eager attention.
- (Phase 4) Computer 1's available RAM (1.45–2.5 GiB during this phase)
  is below what GPT-2 needs on one worker with the default margins.

- Computer 2 and Computer 3 hardware specs are still unverified --
  neither via the Phase 0 PowerShell probe nor via this phase's
  Python hardware profiler. Did not block Phases 1-3; will block
  Phase 4 (partitioning). docs/HARDWARE_PROFILING.md gives the exact
  command to run there.
- `docs/PROJECT_SPEC.pdf` has not been placed yet (checked again
  during Phase 3 -- still absent).
- `go test -race` is unavailable on this machine (no 64-bit C
  toolchain for cgo). Unchanged since Phase 2.
- No worker has ever registered from a non-loopback address; physical
  multi-machine testing has not happened. Unchanged since Phase 2 --
  see docs/NETWORKING.md.
- NVIDIA, AMD and Intel Arc detection, and every CUDA/XPU/DirectML
  "available" path, are tested only with simulated inputs. No such
  hardware exists in this project. Computer 1 exercises only the
  Intel iGPU and the "CUDA build, no GPU" paths.
- `torch_directml` isn't installed on Computer 1, so only its
  "not installed" branch is verified. It was deliberately not
  installed.
- The CPU GFLOPS micro-benchmark spread ~10× between runs (31–294) on
  this loaded machine. It only shows that the backend works, not
  usable throughput (Phase 7).
- Windows CPU frequency never shows turbo; load average is hidden for
  the first 60 s; WMI can time out under heavy load (driver version
  then missing until the next refresh).
- A force-killed worker can leave its backend-probe child running for
  a few seconds (~500 MB). Graceful shutdown cleans it up.
- The machine was running at about 80% RAM use from other software
  during testing, which affected the timing figures above.

## Next implementation steps

- (After Phase 8) Run the first real two-computer test:
  `scripts/phase8_lan.py run … --interactive` (DEPLOYMENT_LAN.md, Step
  7). It must report `REAL_MULTI_MACHINE`; then fill in the results
  table.

- (After Phase 7, on approval) When Computer 2 is available, follow the
  DEPLOYMENT_LAN.md checklist and fill in its results table: first the
  `REAL_MULTI_MACHINE` run, then the disconnect/restart repeats. After
  that: TLS with mutual authentication, and advertising execution
  endpoints through registration.

- (After Phase 6, on approval) Run the two-worker pipeline across
  physical machines once Computers 2/3 are reachable (REMOTE_EXECUTION.md,
  "Private-LAN setup"). Then consider authentication (TLS with mutual
  auth or a shared token), endpoint advertisement through registration,
  and worker-to-worker forwarding instead of hub-and-spoke.
0. (After Phase 5; done in Phase 6) A remote `ActivationTransport`: a
   protobuf message carrying every `ActivationMeta` field plus the raw
   tensor bytes; stages hosted inside the worker process; then the tiny
   model across two worker processes, labelled as such until two
   physical machines are used. The earlier step 1 below describes the
   same direction.
1. (Original Phase 4 note) Phase 5, gRPC tensor transport. Stage boundaries and
   the boundary tensor sizes come from the Phase 4 plan
   (PARTITION_PLAN_SCHEMA.md, "Integration points"). Note that Computer 1
   currently cannot host GPT-2 alone under its desktop load and the
   default margins. Local multi-process work in Phases 5–6 will need
   freed memory or deliberately reduced margins, and must not be
   presented as independent machines.
2. Whenever Computer 2/3 become reachable: register them as REAL
   workers (docs/HARDWARE_PROFILING.md, "Registering Computer 2 and
   Computer 3", including `advertise_host`), repeat Phase 2's
   three-worker scenario across physical machines, apply and test the
   firewall rules in docs/NETWORKING.md, and re-evaluate
   `heartbeat_timeout_seconds` against real measured latency.
3. In parallel, whenever convenient: place docs/PROJECT_SPEC.pdf if it
   exists.
