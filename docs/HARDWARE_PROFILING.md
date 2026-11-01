# Hardware Profiling (Phase 3)

How a worker detects its own hardware and inference capability, how
that reaches the orchestrator, and what the REST API exposes. Field
reference: [HARDWARE_CAPABILITIES.md](HARDWARE_CAPABILITIES.md).
Simulated profiles: [SIMULATED_HARDWARE.md](SIMULATED_HARDWARE.md).

Current hardware schema version: **3** (see "Versioning" below).

## Architecture

```
worker process                                         orchestrator
--------------                                         ------------
HardwareProfiler (worker/hardware/profiler.py)
 ├─ static profile  ── cached; re-collected only on ─┐
 │   CPU identity        startup, every                │  Register(HardwareInfo)
 │   GPU inventory       hardware_refresh_interval,    ├──────────────────────► grpcserver
 │                       and after a reconnection      │                          │ Options: allow_simulated_workers
 │   backend probe   ── once, child process, in the ──┘ (re-register when done)  │
 │                                                     │                          ▼
 └─ dynamic metrics ── every heartbeat ───────────────┘  Heartbeat(metrics)   registry (InMemoryRegistry)
     CPU %, freq, load, RAM, swap       (5 s default)  ───────────────────►     │ ValidateHardware / ValidateLiveMetrics
     GPU counters  ── own slower interval (15 s)                                 │ stores record + timestamps
                                                                                 ▼
                                                       capability.Classify / ComputeResources
                                                                                 ▼
                                                       REST: /api/workers[...], /api/resources, /health
```

Three things are kept separate throughout, as the phase required:

1. **Detected physical hardware** — `gpus`, CPU identity, RAM total.
2. **Runtime capability** — `backend_capabilities`: what PyTorch can
   actually compute on, verified by a small test. A GPU in (1) says
   nothing about (2).
3. **Currently available resources** — live metrics, and the derived
   `resources` view (available minus safety margin).

And a fourth axis on every record: **profile source** — `REAL`
(measured here), `SIMULATED` (a test profile), or `UNKNOWN` (an old
worker that didn't say; never assumed real).

## Static vs. dynamic, and what each costs

Measured on Computer 1:

| What | How | Cost | When |
|---|---|---|---|
| Static inventory | registry read, psutil, nvidia-smi, PowerShell/WMI (~1.2 s), DXGI (~1 ms CPU) | **~1.5 s** | startup; every `hardware_refresh_interval_seconds` (300 s); after reconnecting |
| Backend probe | `import torch` + compute test, **in a child process** | **8–14 s** (the torch import alone: 8.3 s cold, 13.9 s with three workers starting at once) | **once per worker process**, in the background |
| Live CPU/RAM/swap (`collect_live_metrics()`) | psutil only | 0.23 ms CPU | every heartbeat (5 s) |
| GPU counters (`collect_gpu_metrics()`) | in-process PDH query, kept open | ~4 ms CPU per sample (154 ms one-time setup) | every `gpu_metrics_interval_seconds` (15 s) |

Steady state of an idle worker at default intervals: **0.6% of one
core and 53 MiB RSS** (measured over 60 s, after startup).

### Startup: register first, probe in the background

The startup sequence:
1. Collect the static inventory and live metrics (~1.5 s).
2. Register immediately, with `backend_probe_pending: true`.
3. Run the backend probe in a child process
   (`python -m worker.hardware.backend_probe`) on a background thread.
4. When the probe finishes, re-register with the results
   (`backend_probe_pending: false`).

While the probe is pending, the orchestrator classifies the worker as
`UNCLASSIFIED` ("probe still running"), not as `NO_COMPUTE_BACKEND`.
Its resources view lists no execution backends and is flagged
`backend_probe_pending`.

Why it is done this way: an earlier version waited for the probe
before registering, and three workers started together took **23–26 s**
to appear, which broke Phase 2's three-worker test. With the probe
deferred they appear in **4.2–4.6 s**. Running the probe in a child
process also means the worker never keeps PyTorch loaded just to
describe itself: RSS went from 531 MiB to 53 MiB, and a GPU driver
that crashes during the probe takes down only the child.

The probe runs once per process, because what a framework can compute
on doesn't change while the process runs. Periodic refreshes and
reconnections re-collect the inventory and metrics, not the probe.

No subprocess is ever started on the heartbeat path. GPU *enumeration*
(subprocesses) happens only with the static profile; GPU *metrics* use
an in-process Windows API. Tests enforce both:
`test_collect_live_metrics_never_calls_gpu_detection`,
`test_profiler_live_metrics_never_redetect_hardware`,
`test_profiler_caches_static_profile`, `test_profiler_gpu_metrics_respect_interval`.

A heartbeat whose GPU sample isn't due carries an empty `gpu_metrics`
list. The orchestrator reads empty as "not sampled this time" and keeps
the previous values. Every scalar metric is a proto3 `optional`, so
"not reported" is never confused with a real `0`.

## Detection methods

### CPU
- **Manufacturer / model**: Windows registry
  `HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\0`
  (`ProcessorNameString`, `VendorIdentifier`) — the marketing name
  Task Manager shows. Fallback: `platform.processor()`'s CPUID string.
- **Physical vs. logical**: two separate calls,
  `psutil.cpu_count(logical=False)` and `(logical=True)`. Computer 1
  (hybrid P/E cores) correctly reports 10 / 12. If psutil can't report
  physical cores, the value is `0` (unknown); it is never filled in
  with the logical count.
- **Utilization**: `psutil.cpu_percent(interval=None)` (non-blocking,
  since the previous call).
- **Frequency**: `psutil.cpu_freq()`. See the Windows limitations below.
- **Load**: `psutil.getloadavg()` 1-minute value (emulated on Windows).
- **Architecture**: `platform.machine()`.

### RAM and memory pressure
- `psutil.virtual_memory()`: total, available, used, percent. Total
  and available are reported separately. Planning uses **available**
  (see `resources` below).
- `psutil.swap_memory()`: page-file total/used/percent. This is the
  memory-pressure indicator: page-file use rising while RAM is nearly
  full. The page-file total is sent on heartbeats because Windows
  resizes it; on Computer 1 it grew from 14.8 GB to 15.9 GB during
  this phase's testing.

### GPU (three sources, merged)
1. **nvidia-smi** (`--query-gpu=name,memory.total,driver_version`):
   authoritative VRAM and driver for NVIDIA cards. If nvidia-smi is
   absent, no NVIDIA entry is produced; one is never assumed.
2. **WMI via PowerShell** (`Get-CimInstance Win32_VideoController`):
   every display adapter of any vendor, plus driver version. Vendor
   comes from the PCI vendor ID in `PNPDeviceID` (10DE NVIDIA,
   1002/1022 AMD, 8086 Intel), with the device name as a fallback.
3. **DXGI** (`CreateDXGIFactory1` → `EnumAdapters1` → `GetDesc1`,
   called via ctypes in `worker/hardware/windows_gpu.py`; ~13 ms):
   64-bit `DedicatedVideoMemory` and `SharedSystemMemory`, and the
   adapter LUID used as the GPU `id`. The "Microsoft Basic Render
   Driver" software adapter is filtered out.

DXGI enriches the WMI/nvidia-smi entries (matched by name). If WMI
fails entirely (no PowerShell, a query error), DXGI alone still
supplies the inventory. **A detection failure is not reported as "no
GPU"** (`test_wmi_failure_does_not_mean_no_gpu`).

**Dedicated vs. shared.** This is decided by the measured DXGI size,
not by the device's name:
- DXGI dedicated ≥ 1 GiB → `dedicated` (real VRAM).
- An Intel or AMD GPU below that → `shared` (integrated). Its small
  "dedicated" figure is a firmware carve-out of system RAM. It is
  recorded raw in `dedicated_memory_bytes`, but it is not VRAM.
- No DXGI figure: NVIDIA → `dedicated`; Intel → `shared`, except names
  containing "Arc" → `unknown` (Arc is both a discrete line and, since
  Meteor Lake, an integrated brand); AMD → `unknown` (WMI can't tell an
  APU from a discrete card).
- `memory_total_bytes` (schema 2's field) is now set **only for
  dedicated VRAM**. For an integrated GPU it is absent. Schema 2
  reported Iris Xe's 2 GiB WMI aperture there, which this phase
  corrects. The orchestrator rejects a schema-3 profile that reports
  `memory_total_bytes` on a shared GPU.

**Live GPU metrics** (PDH, `windows_gpu.GPUCounterSampler`):
- `\GPU Engine(*)\Utilization Percentage`: summed per adapter and
  engine type, then the busiest engine type, capped at 100. This is
  Task Manager's rule.
- `\GPU Adapter Memory(*)\Dedicated Usage` and `Shared Usage`.

Utilization is a rate, so the first sample after startup has none
(absent), and later samples average the time since the previous one.

### Inference backends (`worker/hardware/backend_probe.py`)
If `import torch` fails, that is the answer: nothing is installed or
probed, and every backend is "undetermined" (`available` absent). If
torch is present, each backend it knows is probed independently, so
one crashing probe can't hide another:

| backend | availability check | device info |
|---|---|---|
| `cpu` | torch importable | torch thread count |
| `cuda` | `torch.cuda.is_available()`; `version` = `torch.version.cuda` (the build's CUDA version) | `get_device_name` |
| `xpu` | `torch.xpu.is_available()` (omitted if the build has no `torch.xpu`) | `get_device_name` |
| `mps` | `torch.backends.mps.is_available()` | — |
| `directml` | `import torch_directml` (separate package) | `device_name` |

**Compute test.** Each available backend gets a 256×256 float32
matmul. The test checks `A @ I == A` (a wrong result fails the test,
not just an exception) and times 8 repetitions to get a rough GFLOPS
figure. No model is downloaded, and it uses well under 1 MB.
`backend_compute_test: false` in `worker.yaml` disables it. A
backend counts as **usable** when it is available and its test didn't
fail.

The GFLOPS figure is a **micro-benchmark**, measured once, while the
machine does whatever else it's doing. On Computer 1 (with RAM about
80% used by other software) it read **31.1, 155.9 and 293.6 in three
runs**, a ~10× spread. A single reading is too noisy for scheduling
decisions. Phase 7 needs repeated or longer measurements, and until
then this number should only be read as "the backend computes".

## Versioning

`hardware_schema_version` on every profile:
- **1**: Phase 1/2 fields only, or unset (0), which is read as 1.
- **2**: + CPU identity, GPU list, torch/CUDA/DirectML flags, live CPU/RAM.
- **3**: + profile source and timestamp, CPU frequency and load, swap,
  `backend_capabilities`, DXGI memory figures, GPU `id`, `gpu_metrics`.

Compatibility rules:
- Every protocol change is additive (new field numbers only). Old
  workers still register, and a hand-built schema-1 object goes over
  the wire as schema 1 with source `UNSPECIFIED`.
- Validation is versioned: rules a schema-2 worker couldn't satisfy
  apply only from schema 3 (`TestValidateHardware_VersionedRules`).
- A schema-3 profile must declare `REAL` or `SIMULATED`.
- Schemas newer than the orchestrator knows
  (`registry.LatestHardwareSchemaVersion`) are accepted; their unknown
  fields are ignored.

## Validation (orchestrator)

`registry.ValidateHardware` / `ValidateLiveMetrics` reject impossible
or self-contradictory data. Examples: physical cores > logical, RAM
available > total, a percentage outside 0–100, negative byte counts, a
non-positive frequency, duplicate GPU ids, `test_passed` set without
`tested`, a SIMULATED profile without a name, a REAL profile carrying
one, and GPU metrics for a GPU id that isn't in the registered profile.

Every problem is listed in the rejection message. An invalid heartbeat
is rejected as a whole and the stored record is left unchanged.
**Incomplete data is not invalid**: unknown stays unknown.

## Reconnection

- **Network outage**: heartbeats fail and the orchestrator marks the
  worker offline after `heartbeat_timeout_seconds`. On the first
  successful heartbeat afterwards, the worker re-collects its full
  profile and re-registers.
- **Orchestrator restart** (in-memory registry lost): the heartbeat
  reply carries `reregister_required`, and the worker re-registers
  with a fresh profile.

Before this phase, a restarted orchestrator never saw an
already-running worker again. Both paths are verified with real
processes in `test_worker_reregisters_with_fresh_profile_after_orchestrator_restart`.

## REST API

All endpoints are read-only GET (other methods return 405). Unknown
values are absent from the JSON, never `0` or `false`.

| Endpoint | Returns |
|---|---|
| `GET /health` | adds `online_real_workers`, `online_simulated_workers`, `online_unknown_source_workers` |
| `GET /api/workers[?source=real\|simulated\|unknown]` | every worker, schema-3 fields + `classification` |
| `GET /api/workers/{id}` | one worker (404 JSON if unknown) |
| `GET /api/workers/{id}/hardware` | static profile only, plus classification |
| `GET /api/workers/{id}/metrics` | dynamic metrics only, with `metrics_updated_at` / `metrics_age_seconds` |
| `GET /api/workers/{id}/resources` | partition-planning view (see HARDWARE_CAPABILITIES.md) |
| `GET /api/resources` | planning view of all workers, in separate `real` / `simulated` / `unknown_source` lists |

Actual output, captured 2026-10-31 07:15 from the final build: the
real orchestrator, Computer 1's real worker and one simulated worker,
over loopback.

`GET /api/workers/computer-1/metrics`
```json
{
  "worker_id": "computer-1", "profile_source": "REAL", "online": true, "status": "idle",
  "metrics_updated_at": "2026-10-31T07:15:21.1678891+05:30", "metrics_age_seconds": 1.06,
  "metrics_collected_at": "2026-10-31T07:15:21.167+05:30",
  "cpu": { "frequency_current_mhz": 1457, "utilization_percent": 14.8 },
  "ram": { "available_bytes": 3636699136, "total_bytes": 16876888064, "used_bytes": 13240188928, "utilization_percent": 78.5 },
  "swap": { "total_bytes": 17398558720, "used_bytes": 3047477247, "utilization_percent": 17.5 },
  "gpus": [ { "gpu_id": "luid:0x00000000_0x000160E5", "utilization_percent": 12.53,
              "dedicated_memory_used_bytes": 0, "shared_memory_used_bytes": 1464348672 } ]
}
```
(`load_average_1m` is absent because the worker had been up for less
than a minute. See the Windows limitations below.)

`GET /api/workers/computer-1/resources`
```json
{
  "worker_id": "computer-1", "online": true, "profile_source": "REAL",
  "ram_total_bytes": 16876888064, "ram_available_bytes": 3636699136,
  "ram_safety_margin_bytes": 1687688806, "ram_usable_bytes": 1949010330,
  "swap_utilization_percent": 17.5,
  "cpu_physical_cores": 10, "cpu_logical_cores": 12, "cpu_utilization_percent": 14.8,
  "cpu_frequency_max_mhz": 1700, "cpu_measured_gflops": 31.05,
  "execution_backends": ["cpu"], "backend_gflops": { "cpu": 31.05 }, "backend_probe_pending": false,
  "gpus": [ { "gpu_id": "luid:0x00000000_0x000160E5", "name": "Intel(R) Iris(R) Xe Graphics",
              "vendor": "Intel", "memory_type": "shared", "compute_usable": false, "utilization_percent": 12.53,
              "note": "integrated GPU: uses system RAM, which is already counted under RAM; no separate VRAM" } ],
  "metrics_age_seconds": 1.07
}
```

`GET /api/workers/computer-1/hardware` during the first seconds,
before the probe finished (an actual capture from an earlier run):
```json
{ "...": "...", "torch_installed": false, "backend_probe_pending": true, "backend_capabilities": [],
  "classification": { "classes": ["UNCLASSIFIED"], "memory_tier": "STANDARD",
    "reasons": ["UNCLASSIFIED: the worker's inference backend probe is still running; compute classes will follow its next registration", "..."] } }
```
(`torch_installed: false` carries no information while
`backend_probe_pending` is true. See HARDWARE_CAPABILITIES.md.)
The full `GET /api/workers/{id}` response for Computer 1 is in
DEVELOPMENT_STATUS.md.

## Known Windows limitations

- **CPU frequency never shows turbo.** On Computer 1 psutil reports
  max = 1700 MHz, the i7-1355U's base clock. "Current" drops below it
  when power-throttled (1457 MHz was observed) but never rises above
  it, so boost clocks are invisible. Reading true per-core clocks needs
  performance counters or a kernel driver; neither is implemented.
- **Load average is emulated.** psutil derives it from the processor
  queue length in a background thread that returns zeros at first. It
  is reported as unknown for the worker's first 60 s, then as psutil
  gives it. It counts queued threads; it is not a percentage.
- **WMI `AdapterRAM` is 32-bit.** It misreports above about 4 GB and
  reports an aperture for iGPUs. DXGI's 64-bit figures replace it
  wherever DXGI can see the adapter. If only WMI is available, it is
  used for dedicated cards and labelled `memory_source: "wmi"`.
- **DXGI's iGPU "dedicated" figure is system RAM.** It is a firmware
  carve-out (128 MiB on Iris Xe). It is recorded but never counted as
  VRAM.
- **PDH GPU counters need Windows 10 1709+ and a WDDM 2.x driver.** If
  they are missing, GPU metrics are absent; the GPU is not removed
  from the inventory.
- **AMD, NVIDIA and Intel Arc paths are tested with simulated inputs
  only.** No such hardware exists in this project yet. AMD has no
  vendor tool integrated (ROCm SMI/ADL), and AMD/Intel GPU compute
  backends (ROCm, XPU builds of torch) are probed but cannot be
  exercised here.
- **DirectML** is probed and compute-tested only if `torch-directml`
  is installed. It isn't on Computer 1, and wasn't installed: no
  unnecessary packages.
- **WMI can time out under heavy load.** The PowerShell query has a 5 s
  timeout. In one run on a heavily loaded Computer 1 it returned
  nothing: DXGI still reported the GPU, but `driver_version` was empty
  for that profile. It returns at the next refresh (300 s).
- **A hard-killed worker can orphan its probe child.** A graceful
  shutdown kills a running probe child. A force-killed worker
  (`Stop-Process`, Task Manager) can't, and on Windows the child
  survives its parent. It finishes the probe (seconds) and exits by
  itself, but meanwhile it uses about 500 MB and competes for CPU.
  Fixing this fully would need a Windows Job Object.
- **The fast first profile takes ~1.5 s** (mostly the WMI query), which
  is why the inventory is cached and refreshed only every 300 s.
- **Non-Windows**: DXGI/PDH are skipped (empty results). nvidia-smi,
  psutil and torch probing still work. This hasn't been exercised,
  since every machine in this project runs Windows.

## Running the profiler

```powershell
cd "C:\Users\Yakob\OneDrive\Documents\Programs\decentralized llm\worker"
python -c "import dataclasses, json; from worker.hardware.profiler import HardwareProfiler; print(json.dumps(dataclasses.asdict(HardwareProfiler().full_snapshot()), indent=2, default=str))"
```

## Adding a new kind of worker hardware

- **New GPU vendor**: add its PCI vendor ID to `_PCI_VENDOR_IDS` in
  `profiler.py`. If it has a vendor tool (like nvidia-smi), add a
  detector, merge it in `_detect_gpus`, and add a monkeypatched test in
  `tests/test_hardware_schema3.py` modelled on the AMD ones.
- **New compute backend**: add a `_probe_<name>` function to
  `backend_probe._PROBES`. `BackendCapability.name` is an open set, so
  no proto change is needed. If it is vendor-specific, add it to
  `backendVendors` in `orchestrator/internal/capability/capability.go`
  so the planner knows which GPU it drives.
- **New field**: add it to `common.proto` with a new field number
  (`optional` if it can be unknown), bump `HARDWARE_SCHEMA_VERSION` /
  `LatestHardwareSchemaVersion` if readers need to know, regenerate
  with `scripts/generate_protos.ps1`, then thread it through
  `registry_client.py` → `grpcserver` → `registry.WorkerRecord` → `api`.

## Registering Computer 2 and Computer 3

Nothing about their hardware is known or assumed; the orchestrator
learns it when they connect. No orchestrator change or per-machine
configuration is needed; workers register dynamically.

1. On Computer 1: run the orchestrator with its normal config
   (`allow_simulated_workers: false`). Open the firewall ports in
   [NETWORKING.md](NETWORKING.md).
2. On Computer 2 or 3: copy the repo, then
   `cd worker; python -m pip install -r requirements.txt`. PyTorch is
   optional; without it the worker registers as `NO_COMPUTE_BACKEND`,
   which is accurate.
3. Create `worker.yaml`:
   ```yaml
   worker_id: "computer-2"                 # stable, unique
   orchestrator_address: "<Computer 1 LAN IP>:9090"
   listen_host: "0.0.0.0"
   listen_port: 50100
   advertise_host: "<this machine's LAN IP>"   # if its hostname doesn't resolve from Computer 1
   simulated_profile: ""                   # REAL hardware
   ```
4. `python -m worker.main --config worker.yaml`
5. On Computer 1: `GET /api/workers/computer-2/hardware` and check that
   `profile_source` is `REAL`. Whatever that shows (CPU split, GPU
   vendor, whether CUDA actually works) is the verified answer for
   that machine. Record it in DEVELOPMENT_STATUS.md, and remove or
   rename any simulated profile that might be confused with it.
