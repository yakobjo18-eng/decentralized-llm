# Hardware Capabilities — Field Reference (Phase 3)

The field-by-field schema for the hardware data that flows: Python
`HardwareSnapshot` (`worker/worker/hardware/profiler.py`) → proto
`HardwareInfo` / `HeartbeatRequest` (`shared/proto/common.proto`,
`registry.proto`) → Go `WorkerRecord`
(`orchestrator/internal/registry/registry.go`) → REST JSON
(`GET /api/workers`). For the design rationale (why live metrics and
the full snapshot are separate, why detection is split from tested
capability), see docs/HARDWARE_PROFILING.md.

## Schema versioning

`hardware_schema_version`: `1` means only the original Phase 1/2
fields below are meaningfully populated (an older worker, or any
object built before Phase 3 -- including every Phase 1/2 test
fixture). Unset / `0` is treated identically to `1` by readers. `2`
adds the "Phase 3 fields (schema version 2)" sections. `3` (current)
adds the "Schema version 3" section, plus capability classification
and the resources view, which the orchestrator derives itself. See
HARDWARE_PROFILING.md, "Versioning", for the compatibility rules.

## Phase 1/2 fields (schema version 1+, always populated)

| Field | Type | "Unknown" representation | Notes |
|---|---|---|---|
| `hostname` | string | n/a, always set | |
| `cpu_logical_cores` | int | `0` | Logical processors, i.e. threads. See `cpu_physical_cores` below -- these are NOT the same number on a hybrid-core CPU. |
| `total_ram_bytes` | int64 | `0` | |
| `has_cuda_gpu` | bool | n/a | Derived: true if any detected GPU has `"CUDA"` in its `backends`. Kept for Phase 1/2 compatibility. |
| `gpu_name` | string | `""` | The first CUDA-capable GPU's name, or empty. |
| `gpu_vram_bytes` | int64 | `0` | That GPU's VRAM, or 0. |
| `os` | string | n/a | e.g. `"Windows 11"` |
| `python_version` | string | n/a | |

## Phase 3 fields (schema version 2)

### CPU

| Field | Type | "Unknown" representation |
|---|---|---|
| `cpu_manufacturer` | string | `"Unknown"` |
| `cpu_model` | string | `"Unknown"` |
| `cpu_physical_cores` | int32 | `0` |
| `cpu_architecture` | string | `""` (e.g. `"AMD64"`) |
| `cpu_utilization_percent` | optional double | absent/omitted |

`cpu_physical_cores` vs. `cpu_logical_cores`: on a hybrid-core CPU
(e.g. this project's Computer 1: 10 physical cores, 12 logical
processors), these are deliberately different numbers. Reporting the
logical count as the physical count is the specific error this
project was told to avoid, and is covered by a dedicated test
(`test_computer_1_ground_truth_not_conflated`).

### RAM

| Field | Type | "Unknown" representation |
|---|---|---|
| `ram_available_bytes` | optional int64 | absent/omitted |
| `ram_used_bytes` | optional int64 | absent/omitted |
| `ram_utilization_percent` | optional double | absent/omitted |

These appear in BOTH `HardwareInfo` (a snapshot at registration/
refresh time) and `HeartbeatRequest` (refreshed every heartbeat). The
registry's stored value is whichever arrived most recently; a
heartbeat that doesn't report them leaves the previous value
untouched (see `HeartbeatWithMetrics` in registry.go) rather than
zeroing it.

### System

| Field | Type |
|---|---|
| `os_name` | string (e.g. `"Windows"`) |
| `os_version` | string (e.g. `"11"`) |
| `os_build` | string (e.g. `"10.0.26200"`) |

Worker ID is NOT part of this hardware payload -- it belongs to the
registration envelope (`RegisterRequest.worker_id`), not the hardware
snapshot itself, to avoid duplicating identity information across two
different messages.

### GPU inventory (`gpus`, repeated `GPUInfo`)

| Field | Type | "Unknown" representation |
|---|---|---|
| `vendor` | string | `"Unknown"` (one of `"NVIDIA"`, `"AMD"`, `"Intel"`, `"Unknown"`) |
| `name` | string | n/a |
| `memory_total_bytes` | optional int64 | absent/omitted -- **never a fabricated 0**. Schema 3: dedicated VRAM only; absent for integrated GPUs |
| `memory_type` | string | `"unknown"` (one of `"dedicated"`, `"shared"`, `"unknown"`) |
| `backends` | repeated string | `[]` (empty array -- e.g. `["CUDA"]` for an NVIDIA GPU with a working driver) |
| `driver_version` | string | `""` |

A CPU-only worker still has a `gpus` field; it's an empty array `[]`,
not an omitted field -- a worker with zero GPUs is a known, reportable
state. Since schema 3 there are three independent GPU sources
(nvidia-smi, WMI, DXGI), so one source failing no longer empties the
list. `[]` now requires every source to find nothing, or all of them
to fail. See `test_wmi_failure_does_not_mean_no_gpu`.

`backends` on a `GPUInfo` is hardware-level ("a driver was found for
this device"), not proof any framework can use it -- see
`torch_installed` / `cuda_available` below for tested capability.

### Inference backend capability

| Field | Type | "Unknown" representation | Meaning |
|---|---|---|---|
| `torch_installed` | bool | n/a (always a real answer) | Whether `import torch` succeeds. Never forced. |
| `torch_version` | string | `""` / omitted | e.g. `"2.10.0+cu126"` |
| `cpu_backend_available` | bool | n/a | True when `torch_installed` is true and, since schema 3, the CPU compute test did not fail |
| `cuda_available` | optional bool | **absent/omitted = unknown** (torch not installed, never tested) | Only ever `true`/`false` once `torch.cuda.is_available()` was actually called |
| `cuda_device_count` | optional int32 | absent/omitted | Only meaningful when `cuda_available` is `true` |
| `directml_available` | optional bool | absent/omitted = unknown | Whether `torch_directml` is importable |

The tri-state distinction here is load-bearing, not cosmetic: a REST
client checking `cuda_available === false` is asking "was this tested
and found absent?", while checking for the key's absence is asking
"was this ever tested at all?" Conflating the two would make an
untested worker look identically "no CUDA" to one that was tested and
confirmed to have none.

## Schema version 3

### Profile identity

| Field | Type | Meaning |
|---|---|---|
| `profile_source` | enum → JSON string | `REAL` (measured on this machine) · `SIMULATED` (a YAML test profile) · `UNKNOWN` (sender predates schema 3; **never assumed REAL**) |
| `profile_notice` | JSON only | `"SIMULATED - NOT PHYSICALLY VERIFIED"` for simulated workers, an explanation for UNKNOWN, absent for REAL |
| `simulated_profile_name` | string | Required for SIMULATED, forbidden for REAL |
| `profile_collected_at` | timestamp | When the worker collected the static profile (worker clock) |
| `hardware_updated_at` | timestamp | When the orchestrator last received a full profile (orchestrator clock) |
| `metrics_updated_at` | timestamp | When live metrics last changed (orchestrator clock; a heartbeat with no metrics doesn't move it) |
| `metrics_collected_at` | timestamp | When the worker sampled those metrics (worker clock) |

### CPU, memory pressure

| Field | Type | "Unknown" representation | Notes |
|---|---|---|---|
| `cpu_frequency_current_mhz` | optional double | absent | Windows: drops when power-throttled, never shows turbo above base |
| `cpu_frequency_max_mhz` | optional double | absent | |
| `cpu_load_average_1m` | optional double | absent | Emulated on Windows; absent for the worker's first 60 s |
| `swap_total_bytes` / `swap_used_bytes` / `swap_utilization_percent` | optional | absent | Page file; heartbeat-refreshed (Windows resizes it) |

### GPU (additions to `GPUInfo`)

| Field | Type | Meaning |
|---|---|---|
| `id` | string | DXGI adapter LUID, `"luid:0x00000000_0x000160E5"`; simulated GPUs use `"sim:<profile>:<n>"` |
| `dedicated_memory_bytes` | optional int64 | DXGI `DedicatedVideoMemory` (nvidia-smi's figure for NVIDIA). For an integrated GPU this is a firmware carve-out of system RAM; it is recorded, but it is not VRAM |
| `shared_memory_bytes` | optional int64 | DXGI `SharedSystemMemory`: the *ceiling* of system RAM the GPU may borrow. A limit, not an allocation, and the same RAM the CPU uses |
| `memory_source` | string | `nvidia-smi` · `dxgi` · `wmi` · `simulated` · `""` |
| `integrated` | bool | true when `memory_type` is `shared` |

Live per-GPU metrics (`GPUMetrics`, matched by `gpu_id`), all optional:
`utilization_percent` (busiest engine type, Task Manager's rule),
`dedicated_memory_used_bytes`, `shared_memory_used_bytes`. In
`GET /api/workers` they are merged into each GPU object.

### Inference backends (`backend_capabilities`, repeated)

| Field | Meaning |
|---|---|
| `name` | `cpu` · `cuda` · `xpu` · `mps` · `directml` · … (open set) |
| `available` | Tri-state: absent = never probed |
| `usable` | JSON only: available and the compute test didn't fail |
| `device_count`, `devices` | As the framework reports them |
| `version` | e.g. `"12.6"` for cuda: the version torch was **built** against, not proof a GPU exists |
| `tested`, `test_passed`, `test_gflops` | The 256×256 verified matmul; `test_passed` absent unless tested |
| `detail` | Why unavailable, or the test error |

`backend_probe_pending` (bool) is true from a worker's first
registration until its background backend probe reports, typically
8–14 s on Computer 1. **While it is true, `torch_installed`,
`cpu_backend_available` and `backend_capabilities` carry no
information** (they read false/empty only because nothing has been
probed yet), and the worker is classified `UNCLASSIFIED`. See
HARDWARE_PROFILING.md, "Startup".

`torch_cuda_build_version` duplicates `torch.version.cuda` at the top
level. The schema-2 flags `cuda_available`, `cuda_device_count` and
`directml_available` are still populated, derived from these entries.

## Capability classification (orchestrator-derived)

`classification` on every worker (`orchestrator/internal/capability`):

| Class | Assigned when |
|---|---|
| `GPU_ACCELERATED` | some non-CPU backend is usable |
| `CPU_ONLY` | the CPU backend is usable and no accelerator is. **A present but unusable GPU still means CPU_ONLY** (Computer 1) |
| `NO_COMPUTE_BACKEND` | backends were reported and none works (e.g. no PyTorch) |
| `UNCLASSIFIED` | schema 1 (nothing to classify from), or `backend_probe_pending` |
| `HIGH_MEMORY` / `LOW_MEMORY` | installed RAM ≥ `high_memory_threshold_gib` (32) / < `low_memory_threshold_gib` (8) |
| `MIXED_HARDWARE` | GPUs from more than one vendor, or more than one usable accelerator backend |

Also returned: `memory_tier` (`LOW`/`STANDARD`/`HIGH`/`UNKNOWN`),
`accelerator_backends`, and `reasons`, one line per decision quoting
the reported values. Classification uses installed RAM, because it is
a stable property of the machine. Planning uses *available* RAM (see
below). Classes are labels for filtering and display; they don't
replace the numbers. Thresholds live in `orchestrator.yaml` under
`capability:`.

## Partition-planning resources (orchestrator-derived)

`GET /api/workers/{id}/resources` and `GET /api/resources` (separate
`real`/`simulated`/`unknown_source` lists, never totalled together):

| Field | Derivation |
|---|---|
| `ram_safety_margin_bytes` | max(`ram_safety_margin_percent` (10%) × total, `ram_safety_margin_min_gib` (1 GiB)) |
| `ram_usable_bytes` | max(0, **available** − margin); absent if available is unknown, never falling back to total |
| `execution_backends` | usable backends, `cpu` first |
| `backend_gflops`, `cpu_measured_gflops` | from the compute micro-test |
| `gpus[].vram_*` | dedicated GPUs only: total, used, margin (10% / ≥ 0.5 GiB), usable = total − used − margin. Usable is absent if usage is unknown |
| `gpus[].compute_usable` | a usable backend targets this vendor (cuda→NVIDIA, rocm→AMD, xpu→Intel, directml→any) |
| `total_usable_vram_bytes` | sum over compute-usable GPUs only; absent if none |
| `metrics_age_seconds` | how stale the dynamic figures are |

Integrated GPUs get no VRAM figures: their memory is the system RAM
already counted above.

## REST API (`GET /api/workers`)

Same fields as above, as JSON, with Go's `encoding/json` `omitempty`
applied to every field that can be genuinely unknown (the `optional`
proto fields' Go pointer types) -- so a client can distinguish
"present and false/zero" from "key absent, meaning unknown" by
checking for the key, not just its value. `GET /health` additionally
reports `online_workers` (workers within `heartbeat_timeout_seconds`
of their last heartbeat) alongside the Phase 1 `registered_workers`
count, and since schema 3 splits the online count by profile source.
Full endpoint list: HARDWARE_PROFILING.md, "REST API".

## What is explicitly NOT in this schema (by design, not oversight)

- **Live CPU clock speed** -- not exposed by psutil on Windows (see
  HARDWARE_PROFILING.md).
- **Sustained throughput benchmarks** -- only the micro-test above.
  Real per-layer timing belongs to Phase 7.
- **Network latency/jitter** between nodes -- Phase 7
  ("Latency-Aware Orchestration"), not hardware profiling.
- **Model memory requirements** -- Phase 4 (model partitioning).

(GPU utilization, listed here as not attempted in the first Phase 3
pass, is now implemented via in-process PDH counters.)
