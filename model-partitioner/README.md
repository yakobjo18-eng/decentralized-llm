# model-partitioner

Phase 4: a hardware-aware partitioning engine for sequential
Transformer models. It inspects a real Hugging Face model (config plus
the meta device, with **no weights downloaded or loaded**), estimates the
memory each contiguous range of layers needs, and assigns ranges to
heterogeneous workers so that the slowest pipeline stage is as fast as
possible.

Phase 5 adds **minimal local execution** (`partitioner/execution/`). It
takes a validated plan for the built-in test model
`builtin:tiny-gpt2-v1` and runs real inference in **one process**:

- on this machine's single REAL worker; or
- as a labelled emulation of a multi-stage plan.

Phase 6 adds **remote stage execution**: `execute --remote` sends
each stage to its assigned worker process's StageExecution service over
gRPC (unauthenticated; tested so far with separate processes on one
computer, not across machines). There is still no GPT-2 execution and
no quantized execution. GPT-2 remains plannable only.

Docs: [MODEL_PARTITIONING.md](../docs/MODEL_PARTITIONING.md) ·
[MEMORY_ESTIMATION.md](../docs/MEMORY_ESTIMATION.md) ·
[PARTITION_PLAN_SCHEMA.md](../docs/PARTITION_PLAN_SCHEMA.md) ·
[EXECUTION.md](../docs/EXECUTION.md) (Phase 5) ·
[REMOTE_EXECUTION.md](../docs/REMOTE_EXECUTION.md) (Phase 6)

```
partitioner/
  adapters/      architecture adapters (GPT-2 only); base.py = the interface
  inspection.py  config -> meta-device model -> ModelSpec; checkpoint-header cross-check
  model_spec.py  architecture-neutral spec: ordered units made of components
  memory.py      per-stage memory, every item MEASURED/CALCULATED/ESTIMATED/UNKNOWN
  workers.py     Phase 3 resources: live orchestrator, simulated YAML, or this machine
  cost.py        FLOPs per unit, per-worker throughput and capacity
  optimizer.py   exact DP: min bottleneck, then min total
  api.py         create_plan(); validates its own output; sets `executable`
  validate.py    independent plan validator (+ estimator_for_plan)
  plan.py        schema v1 helpers
  cli.py         python -m partitioner ...
  execution/     Phase 5
    interface.py   versioned stage interface: StageSpec, Activation, StageExecutor, ActivationTransport
    tiny_gpt2.py   builtin:tiny-gpt2-v1: deterministic weights, explicit math, reference forward, stage executor
    models.py      registry of executable models; the planner's `executable` rule
    loader.py      load + validate a plan for execution (real / local-emulation / remote modes)
    limits.py      batch / sequence / activation / weight limits, checked before allocation
    runner.py      execute_plan() -> structured result (local modes)
    wire.py        Phase 6 tensor codec (raw-le-c-v1), validated before allocation
    remote.py      Phase 6 coordinator: execute_plan_remote() over gRPC, bounded safe retries
examples/        plans generated on Computer 1 (2026-10-31)
tests/           pytest (291 tests: 113 Phase 4 + 87 Phase 5 + 91 Phase 6)
```

## Quick start

Uses the worker's Python environment (`../worker/requirements.txt`) plus
`torch` and `transformers` (see `requirements.txt`). Nothing new is
needed for Phase 5. Run from this directory:

```powershell
python -m partitioner inspect --model gpt2
python -m partitioner memory  --model gpt2 --dtype fp16
python -m partitioner plan    --model gpt2 --simulated ../worker/profiles/simulated/*.yaml --out plan.json
python -m partitioner plan    --model gpt2 --orchestrator http://<COMPUTER-1-IP>:8080 --out plan.json
python -m partitioner validate plan.json --model gpt2
python -m pytest
```

```python
from partitioner import inspect_model, create_plan, from_simulated_profile, Precision
spec = inspect_model("gpt2")
plan = create_plan(spec, [from_simulated_profile("../worker/profiles/simulated/profile_a_cpu_worker.yaml")],
                   precision=Precision("fp16"))
```

## Executing a plan (Phase 5)

```powershell
# Real, single worker, this machine.
python -m partitioner plan    --model builtin:tiny-gpt2-v1 --local computer-1 --batch 2 --out tiny.json
python -m partitioner execute tiny.json --local computer-1 --tokens "1,2,3,4,5,6,7,8;9,10,11,12,13,14,15,16"

# Multi-stage plan: every stage in this process, labelled LOCAL EMULATION (not distributed).
python -m partitioner plan    --model builtin:tiny-gpt2-v1 --simulated ../worker/profiles/simulated/*.yaml --require-all-workers --out sim.json
python -m partitioner execute sim.json --emulate-stages-locally --tokens "5,6,7"
```

- `execute` prints a JSON result: status, model, plan id, worker
  assignments, per-stage shapes and timings, output shape, next-token
  ids, the reference check, and any error with its cause chain.
- Exit codes: 0 succeeded, 1 rejected or failed, 2 usage error.

**Computer 1:** under the default Phase 3 RAM margin (10%, minimum
1 GiB), Computer 1 usually has too little usable RAM even for the tiny
model. Every stage carries the measured ~700 MiB torch + transformers
overhead. Pass the same `--ram-margin-percent` / `--ram-margin-min-gib`
to both `plan` and `execute` to use a smaller margin deliberately. See
[EXECUTION.md](../docs/EXECUTION.md).

Remote (Phase 6), one worker process per stage. See
[REMOTE_EXECUTION.md](../docs/REMOTE_EXECUTION.md) for worker setup and
security:

```powershell
python -m partitioner execute plan.json --remote --tokens "1,2,3" --endpoint w-a=127.0.0.1:50201 --endpoint w-b=127.0.0.1:50202
```

```python
from partitioner.execution import execute_plan, REAL_MODE
result = execute_plan("tiny.json", [[1, 2, 3]], mode=REAL_MODE,
                      workers={"computer-1": record}, local_worker_id="computer-1")
```
