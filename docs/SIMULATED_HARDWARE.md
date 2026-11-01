# Simulated Hardware Profiles (Phase 3)

> **SIMULATED - NOT PHYSICALLY VERIFIED.** Everything a simulated
> profile reports was typed into a YAML file. None of it is a
> measurement, and none of it describes Computer 2 or Computer 3,
> whose hardware is unknown.

## Why they exist

Only Computer 1 is physically available. Later phases need
heterogeneous workers to develop against: partitioning (4),
hardware-aware scheduling (7), and failure handling (8). A simulated
profile lets a real worker process (real gRPC, real heartbeats,
real REST exposure) report *hypothetical* hardware. Only the hardware
numbers are fake. The process, the protocol and the orchestrator
logic are all real.

Three local processes share Computer 1's CPU, RAM and network stack.
A cluster of simulated workers is useful for testing algorithms and
protocol behaviour. It is **not** evidence about three-machine
performance, and must never be reported as such.

## How simulated workers are kept apart from real ones

| Safeguard | Where |
|---|---|
| Profile files must say `profile_source: SIMULATED`; anything else (including `REAL`) is rejected | `worker/hardware/simulated.py` |
| Every snapshot built from a profile is stamped `SIMULATED` with the profile's name, and sent as `PROFILE_SOURCE_SIMULATED` | `simulated.py`, `registry_client.py` |
| A simulated worker never inspects the real machine (no psutil, WMI, DXGI or torch calls) | `HardwareProfiler(simulated_snapshot=...)`, `test_simulated_profiler_never_touches_real_hardware` |
| Backends are declared, never "tested": `tested: false`, and `detail` carries the notice | `simulated.py` |
| The orchestrator **rejects** simulated registrations unless `allow_simulated_workers: true` | `grpcserver.Options`, `test_simulated_worker_rejected_unless_allowed` |
| REST marks them with `profile_source: "SIMULATED"` and `profile_notice: "SIMULATED - NOT PHYSICALLY VERIFIED"` | `api/rest.go` |
| `/health` counts them separately; `/api/resources` lists them separately; `/api/workers?source=real` excludes them | `api/rest.go` |
| The worker logs a WARNING at startup naming the profile | `worker/main.py` |
| The orchestrator logs a WARNING at startup when simulated workers are allowed | `cmd/orchestrator/main.go` |
| Each example file starts with the notice and states it is not Computer 2/3 | `test_shipped_example_files_state_they_are_not_computers_2_and_3` |

## Example profiles (`worker/profiles/simulated/`)

These are examples only. Edit any value.

| File | Name | CPU (phys/log) | RAM | GPU | Classification it produces |
|---|---|---|---|---|---|
| `profile_a_cpu_worker.yaml` | `profile-a-cpu-worker` | 8 / 16 | 16 GiB | none | `CPU_ONLY`, STANDARD memory |
| `profile_b_gpu_worker.yaml` | `profile-b-gpu-worker` | 6 / 12 | 32 GiB | NVIDIA-class, 8 GiB dedicated, `cuda` | `GPU_ACCELERATED`, `HIGH_MEMORY` |
| `profile_c_high_memory_worker.yaml` | `profile-c-high-memory-worker` | 16 / 24 | 64 GiB | Intel integrated (shared, no compute backend) | `CPU_ONLY`, `HIGH_MEMORY` |

Validate them, or see what one produces:

```powershell
cd worker
python -m worker.hardware.simulated validate profiles/simulated/*.yaml
python -m worker.hardware.simulated show profiles/simulated/profile_b_gpu_worker.yaml
```

## Format (`profile_format_version: 1`)

```yaml
profile_format_version: 1          # required, must be 1
profile_source: SIMULATED          # required, must be exactly SIMULATED
name: my-test-worker               # required: lowercase, digits, dashes
description: "free text"
hostname: sim-my-test-worker       # default: sim-<name>
os: {name: Windows, version: "11", build: "simulated"}

cpu:                               # required
  manufacturer: Intel
  model: "Simulated CPU"           # required
  architecture: AMD64
  physical_cores: 8                # required, >= 1
  logical_cores: 16                # required, >= physical_cores
  max_frequency_mhz: 3600          # optional, > 0

ram:                               # required
  total_gib: 16                    # required, > 0
  swap_total_gib: 8                # optional

gpus:                              # optional list
  - vendor: NVIDIA                 # NVIDIA | AMD | Intel | Unknown
    model: "Simulated GPU"
    memory_type: dedicated         # dedicated | shared
    vram_gib: 8                    # required for dedicated; FORBIDDEN for shared
    shared_gib: 16                 # optional, <= ram.total_gib
    compute_backends: [cuda]       # cuda(NVIDIA) rocm(AMD) xpu(Intel) directml/vulkan(any)

backends:
  torch_installed: true            # false -> NO_COMPUTE_BACKEND
  torch_version: "simulated"
  cpu: true

dynamic:                           # what every heartbeat reports
  cpu_utilization_percent: 10      # 0-100
  ram_available_gib: 12            # <= ram.total_gib
  swap_used_gib: 0.5               # <= ram.swap_total_gib
  gpus:                            # in the same order as gpus above
    - utilization_percent: 3
      vram_used_gib: 0.6           # <= that GPU's vram_gib
      shared_used_gib: 0
```

Validation is strict. Unknown keys are errors, so a typo can't
silently fall back to a default. Every problem in a file is reported
at once. Combinations no real driver provides (CUDA on an Intel GPU,
VRAM on a shared-memory GPU) are rejected.

## Running a simulated worker

Orchestrator (development only):
```yaml
# orchestrator.yaml
allow_simulated_workers: true      # or ORCHESTRATOR_ALLOW_SIMULATED_WORKERS=true
```

Worker:
```yaml
# sim-b.yaml
worker_id: "sim-b"
orchestrator_address: "127.0.0.1:9090"
listen_host: "127.0.0.1"
listen_port: 50201                 # each local worker needs its own port
simulated_profile: "profiles/simulated/profile_b_gpu_worker.yaml"
```
```powershell
cd worker
python -m worker.main --config sim-b.yaml
```

Then `GET /api/workers?source=simulated`, or `GET /api/resources`
(look under `"simulated"`).

## Using profiles in tests and algorithms (no processes needed)

```python
from worker.hardware.simulated import load_profiles_dir
for p in load_profiles_dir("profiles/simulated"):
    p.snapshot   # HardwareSnapshot (profile_source == "SIMULATED")
    p.live       # LiveMetrics from the dynamic: section
```

Phase 4/7 algorithm tests can build heterogeneous clusters from these
directly. Any result produced that way must be labelled as coming from
simulated hardware.

## When Computers 2 and 3 become available

Run real workers on them (see HARDWARE_PROFILING.md, "Registering
Computer 2 and Computer 3"). Their profiles will be `REAL` and
measured. Simulated profiles stay available for algorithm tests, but
no simulated profile should be renamed or edited to "match" a real
machine and then reported as that machine.
