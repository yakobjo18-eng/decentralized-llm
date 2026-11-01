# Remote stage execution (Phase 6)

Phase 6 runs a plan's stages on **separate worker processes**, with
activations passed between them over gRPC. A coordinator sends each
stage, in pipeline order, to the worker the plan assigns it to. That
worker runs only its own unit range and returns the output, which the
coordinator sends to the next stage's worker.

> **Security, up front.** The protocol is **unauthenticated and
> unencrypted**: plain gRPC on an insecure channel, no TLS, no tokens.
> Worker ids, hostnames and plans in requests are *claims*, not proofs.
> Putting it on a private network limits who can reach it but provides
> **no cryptographic security**. Run it only on a network where you
> trust every machine. See [Security](#security).

Related: [EXECUTION.md](EXECUTION.md) (Phase 5: the model, local modes,
limits) · [PARTITION_PLAN_SCHEMA.md](PARTITION_PLAN_SCHEMA.md) ·
[NETWORKING.md](NETWORKING.md).

## What was verified, and where

| | Status |
|---|---|
| Two- and three-stage pipelines across **separate worker processes on one computer** (Computer 1, 127.0.0.1), real gRPC | **Tested** |
| Full `worker.main` workers (registered with the real Go orchestrator, heartbeating) executing a two-stage pipeline | **Tested** (one computer) |
| Stages on **separate physical computers** over a LAN | **Not tested** (`PENDING`). Computers 2 and 3 are still unavailable to this project. Phase 7 prepared the procedure: [DEPLOYMENT_LAN.md](DEPLOYMENT_LAN.md). |
| Phase 7 operational checks (worker stop/restart, hard kill, orchestrator restart, timeout, malformed requests, limits, clean shutdown) with 3 `worker.main` processes | **Tested** on one computer (`REAL_MULTI_PROCESS`); [report](validation/phase7-local-2026-11-01-run2/report.md) |

Results report what the workers say about themselves: their hostnames
and pids, plus `distinct_worker_processes` and
`distinct_worker_hostnames`. On Computer 1 every result shows **one
hostname**, and `distributed_note` states that this was not a
multi-machine run.

## Architecture

```
coordinator (python -m partitioner execute --remote, or execute_plan_remote())
  │ 1. load + validate the plan (Phase 5 loader, "remote" mode); check input + limits
  │ 2. for stage k = 0..n-1, in order:
  │      ExecuteStageRequest(plan, stage k, activation k) ──gRPC──► worker of stage k
  │      ◄────────────── ExecuteStageResponse(status, activation k+1, timing, error)
  │ 3. compare the final logits with the local Phase 5 reference
  ▼
worker process: StageExecution server (its own port, separate from Health)
   checks version, identity, deadline, plan, assignment, tensor, limits
   → decodes → runs units [start, end) with only those weights → encodes
```

- **Hub and spoke.** Every activation goes coordinator → worker →
  coordinator → next worker. Workers never call each other, and no
  activation passes through shared memory: every hop is a gRPC message.
- **Stateless stages.** There is no KV cache. Workers cache the
  validated plan (by plan id) and the built stage weights (by plan id
  and stage), which carry no per-request state.
- **Code locations:**
  - protocol: [shared/proto/stage_execution.proto](../shared/proto/stage_execution.proto);
  - tensor codec: `model-partitioner/partitioner/execution/wire.py`;
  - coordinator: `.../execution/remote.py`;
  - worker service: `worker/worker/execution/` (`service.py`,
    `server.py`, `__main__.py`).
- **The orchestrator is unchanged.** Registration and heartbeats are
  untouched, and workers do not advertise their execution endpoint, so
  the coordinator is given endpoints explicitly (`--endpoint`). Go stubs
  for the new proto are generated (`orchestrator/internal/proto/execution`)
  but not used yet.

## Protocol: `decentralizedllm.execution.v1.StageExecution`

Two RPCs:
- `ExecuteStage(ExecuteStageRequest) returns (ExecuteStageResponse)`
  runs one stage.
- `GetInfo(GetInfoRequest) returns (GetInfoResponse)` (Phase 7) is
  read-only self-description for preflight. It returns protocol
  versions, worker id, hostname and pid, executable models, encodings,
  and every limit, plus `authenticated: false`. It executes nothing and
  takes no concurrency slot. It was added without a version bump: a
  Phase 6 worker answers `UNIMPLEMENTED`, and preflight then falls back
  to an `ExecuteStage` with protocol version 0, which is rejected before
  anything else is looked at.

The protocol version is `PROTOCOL_VERSION_1`. A worker answers a request
with any other version with `REJECTED` / `PROTOCOL_VERSION`, and states
its own version in the response.

**Request fields:**
- `protocol_version`, `request_id`;
- `plan_id`: the plan's `inputs_fingerprint`;
- `plan_json`: the full plan, UTF-8 JSON;
- `model_id`, `config_fingerprint`;
- `target_worker_id`, `stage_index`, `unit_start`, `unit_end`;
- `input` (TensorPayload);
- `coordinator`: informational only.

**Response fields:**
- `protocol_version`, `request_id`, `worker_id`, `stage_index`, `status`;
- `output`: TensorPayload, set only when OK;
- `timing`: validate, compute and total seconds;
- `error`: `code`, `message`, `problems[]`, `executed`;
- `worker_hostname`, `worker_pid`: self-reported.

| `status` | Meaning | Executed? | Coordinator retries? |
|---|---|---|---|
| `OK` | output attached | yes | n/a |
| `REJECTED` | invalid request: version, fields, worker, plan, assignment, tensor, limits, deadline | **no** | no (it would fail again) |
| `BUSY` | at the concurrency limit | **no** | **yes** (bounded) |
| `FAILED` | the stage raised during execution | yes | no |
| `DEADLINE_EXCEEDED` | finished after the deadline; output discarded | yes | no |

Error codes:
- `PROTOCOL_VERSION`, `MALFORMED_REQUEST`, `WRONG_WORKER`, `DEADLINE`;
- `PLAN_INVALID`, `UNASSIGNED_STAGE`, `STAGE_MISMATCH`;
- `INVALID_TENSOR`, `LIMIT_EXCEEDED`;
- `BUSY`, `EXECUTION_ERROR`, `INTERNAL`.

### Tensor format: `raw-le-c-v1`

A `TensorPayload` has these fields:
- `encoding`, `dtype` (enum), `shape` (repeated int64);
- `data` (bytes), `kind` (enum), `produced_by_stage`, `positions_start`.

`data` holds the elements in **C (row-major) order**, each
**little-endian**. There is no header, padding or compression, and
`len(data) == product(shape) × itemsize`. The format is deterministic: a
tensor always encodes to the same bytes.

| dtype | Use | itemsize |
|---|---|---|
| `TENSOR_DTYPE_FLOAT32` (IEEE-754 binary32) | hidden states, logits | 4 |
| `TENSOR_DTYPE_INT64` (two's complement) | token ids | 8 |

Nothing else is accepted. Decoding is `numpy.frombuffer` on validated
bytes, followed by a copy: **no pickle, no eval, no Python objects**.

These checks run before any tensor memory is allocated:
- encoding;
- dtype and kind enums;
- rank 1–3, every dimension > 0;
- `positions_start == 0`;
- byte length equals shape × itemsize;
- total bytes ≤ the limit.

Values are checked right after decoding: fp32 values must be finite.

## What a worker checks, in order

1. **A free concurrency slot**; otherwise `BUSY`.
2. **The request itself:**
   - protocol version, then required fields and a sane unit range;
   - `target_worker_id` is this worker's configured id (else
     `WRONG_WORKER`);
   - a deadline is set, with at least `execution_min_deadline_seconds`
     left (else `DEADLINE`).
3. **The plan:**
   - `plan_json` parses and hashes to `plan_id`;
   - the Phase 5 loader in remote mode accepts it (checked once per plan
     id, then cached): fingerprint, feasible, executable model, fp32, no
     quantization, `validate_plan` with memory recomputed, no SIMULATED
     worker in any stage.
4. **The assignment.** Plan stage `stage_index` must be assigned to
   **this worker id** (else `UNASSIGNED_STAGE`). It must also be on
   **this hostname**, REAL, with the requested unit range, model id and
   config fingerprint (else `STAGE_MISMATCH`).
5. **The tensor, against the stage, before decoding:**
   - kind (`token_ids` for stage 0, `hidden_states` otherwise);
   - dtype, rank, hidden size;
   - `produced_by_stage == stage_index − 1`.
6. **Limits, before decoding:**
   - input bytes;
   - batch size (worker limit and the plan's `workload.batch_size`);
   - tokens (worker limit, model positions, the plan's
     `workload.prompt_length`);
   - **output** bytes, computed from the shape.
7. **Decode and run.** Decode with the finite check, then run the stage:
   its own components' weights only, plus a copy of a tied weight whose
   owner is elsewhere.
8. **Deadline again.** A result finished after the deadline is discarded
   (`DEADLINE_EXCEEDED`).

## What the coordinator does

- **Before sending anything**, it validates the plan (remote mode: no
  SIMULATED workers; worker records are optional, and it warns if they
  are absent). It also requires an endpoint for every stage worker and
  checks the input and limits as in Phase 5.
- **For each response**, it checks:
  - protocol version, request id and stage index;
  - that the **responding worker id is the assigned one**;
  - the output tensor: decoded under the size limit, then kind, shape,
    dtype, producer, request id and finite values, as in Phase 5.
- **At the end**, it compares the final logits with the Phase 5 local
  reference (`reference_forward`), with `REMOTE_REFERENCE_ATOL = 1e-4`.
  Observed on Computer 1: **0.0** (bit-identical, since raw fp32 bytes
  are not rounded in transit). The 1e-4 bound allows for different CPUs
  or BLAS builds ordering fp32 sums differently on different machines.

### Retry policy

A stage is retried, up to `max_attempts` (default 3, maximum 10) with
linear backoff, **only when it is known not to have executed**:

| Situation | Retried? |
|---|---|
| Channel did not become ready within `connect_timeout_seconds`; the request was never sent | yes |
| Worker answered `BUSY` | yes |
| gRPC `RESOURCE_EXHAUSTED`: server-side concurrency cap or message too large, refused before the handler | yes (pointless for a too-large message, but harmless and bounded) |
| gRPC `DEADLINE_EXCEEDED`, `UNAVAILABLE` mid-call, `INTERNAL`, … | **no**: reported as `AMBIGUOUS_FAILURE`, with `executed: null` |
| `REJECTED`, `FAILED`, `DEADLINE_EXCEEDED` from the worker | **no** |

Stages are stateless today, so replaying one would be harmless. The
policy is deliberately strict for when stages hold state (a KV cache).
Every attempt is listed in the result. There is no fault-tolerant
rescheduling: if a stage fails, the request fails.

## Configuration (worker)

These are `worker.yaml` keys, with env overrides where listed:

| Key | Default | Env |
|---|---|---|
| `execution_enabled` | `false` | `WORKER_EXECUTION_ENABLED` |
| `execution_listen_host` | `127.0.0.1` | `WORKER_EXECUTION_LISTEN_HOST` |
| `execution_listen_port` | `50200` | `WORKER_EXECUTION_LISTEN_PORT` |
| `execution_allow_all_interfaces` | `false` | `WORKER_EXECUTION_ALLOW_ALL_INTERFACES` |
| `execution_max_tensor_mib` | 16 | |
| `execution_max_message_mib` | 32 | |
| `execution_max_concurrent` | 2 | |
| `execution_max_batch` | 8 | |
| `execution_max_sequence` | 0 (= model maximum) | |
| `execution_min_deadline_seconds` | 0.05 | |

**Bind policy** (`worker/config/bind_policy.py`; since Phase 7 it is
shared with the worker's Health server, and the Go orchestrator applies
the same rules, see DEPLOYMENT_LAN.md "Security posture"):
- Allowed: loopback and private or link-local IPs.
- `0.0.0.0` / `::`: refused unless `execution_allow_all_interfaces: true`,
  and logged as a warning even then.
- Refused: public (globally routable) IPs, and hostnames (their scope
  can't be checked).
- A worker running a SIMULATED hardware profile refuses to start the
  service.

**Server limits:**
- `grpc.max_receive_message_length` and `grpc.max_send_message_length`
  are set to `execution_max_message_mib`.
- The thread pool has `max_concurrent + 2` threads, and gRPC's
  `maximum_concurrent_rpcs` is `max_concurrent + 2`, so calls beyond
  `max_concurrent` get a quick `BUSY` and anything past that gets
  `RESOURCE_EXHAUSTED`.

Two ways to run the service:
- `python -m worker.main --config worker.yaml` with
  `execution_enabled: true` runs registration, heartbeats, Health **and**
  StageExecution. StageExecution uses its own server and port; the
  Health address is still what the worker registers.
- `python -m worker.execution --config worker.yaml [--port 0]` runs only
  StageExecution, with no registration. Once listening, it prints
  `STAGE_EXECUTION_LISTENING <host>:<port> worker_id=<id>`.

The service preloads the model spec at startup (importing torch and
transformers takes about 11 s on Computer 1), so the first request isn't
slowed by imports.

## Private-LAN setup (not yet exercised on real separate machines)

> **Phase 8:** the procedure now has config generators, firewall and
> port-owner checks, and an automated test run
> (`scripts/phase8_lan.py run`) that classifies by evidence. See
> DEPLOYMENT_LAN.md.
>
> **Phase 7:** the complete, copy-pasteable two-computer procedure is in
> [DEPLOYMENT_LAN.md](DEPLOYMENT_LAN.md). It covers firewall rules
> limited to the peer IP, `preflight`, a physical test checklist, and a
> results table that is `PENDING`. The short version below remains
> correct.

1. **On each worker machine**, find its private LAN IP (`ipconfig`, e.g.
   `192.168.1.21`) and set:
   ```yaml
   worker_id: "computer-2"
   execution_enabled: true
   execution_listen_host: "192.168.1.21"   # private LAN IP, NOT 0.0.0.0
   execution_listen_port: 50200
   ```
2. **Allow the port only from the LAN, on a Private network profile.**
   Run in an elevated PowerShell. `-RemoteAddress LocalSubnet` keeps the
   rule to the local subnet:
   ```powershell
   New-NetFirewallRule -DisplayName "decentralized-llm StageExecution" -Direction Inbound -Protocol TCP `
     -LocalPort 50200 -Action Allow -Profile Private -RemoteAddress LocalSubnet
   ```
   Never forward this port on a router or expose it to the internet.
   Remove the rule when you are done:
   `Remove-NetFirewallRule -DisplayName "decentralized-llm StageExecution"`.
3. **Plan with REAL worker records whose hostnames are the real
   machines' hostnames.** Workers refuse stages planned for another host.
   For example, take the records from the orchestrator with
   `--orchestrator`, which needs enough usable RAM under the Phase 3
   margins; see EXECUTION.md.
4. **Run the coordinator** from any machine that can reach the workers:
   ```powershell
   python -m partitioner execute plan.json --remote --tokens "1,2,3,4" `
     --endpoint computer-2=192.168.1.21:50200 --endpoint computer-3=192.168.1.22:50200
   ```
   Then check `distinct_worker_hostnames` in the result.

## CLI

```
python -m partitioner execute PLAN.json --tokens "1,2,3;4,5,6" --remote \
    --endpoint W1=HOST:PORT --endpoint W2=HOST:PORT \
    [--stage-timeout 30] [--connect-timeout 3] [--max-attempts 3] \
    [--max-batch 8] [--max-seq N] [--max-activation-mib 64] [--no-reference-check] [--include-logits] [--out result.json]
```

- `--remote`, `--local` and `--emulate-stages-locally` are mutually
  exclusive.
- Exit codes are as in Phase 5: 0 succeeded, 1 rejected or failed, 2
  usage error (bad or duplicate `--endpoint`, mode conflicts).

Try it on one machine with two service-only processes:
```powershell
cd worker
python -m worker.execution --config a.yaml --port 50201   # a.yaml: worker_id: "w-a"
python -m worker.execution --config b.yaml --port 50202   # b.yaml: worker_id: "w-b"
```
Then plan with records for `w-a` and `w-b` (REAL, this hostname). The
tests build these with `real_local_worker`; a CLI plan with `--local`
creates only one record. Run `execute --remote` with
`--endpoint w-a=127.0.0.1:50201 --endpoint w-b=127.0.0.1:50202`.

## Result additions (over Phase 5)

| Field | Meaning |
|---|---|
| `execution_mode` | `remote` |
| `distributed`, `distinct_worker_processes`, `distinct_worker_hostnames`, `distributed_note` | `distributed` is true when the stages ran in more than one worker process. Hostnames show whether that was more than one machine, as self-reported. |
| `test_classification`, `test_classification_basis` (Phase 7) | `REAL_MULTI_MACHINE` if 2+ distinct reported hostnames, `REAL_MULTI_PROCESS` if one host with more than one process, `REAL_SINGLE_MACHINE` if everything ran in the coordinator's process. Derived from self-reports (unauthenticated). |
| `protocol_version`, `transport`, `coordinator`, `remote_options` | how it ran |
| `stages[].endpoint`, `worker_reported`, `attempts[]`, `round_trip_seconds`, `worker_timing`, `request_bytes`, `response_bytes` | per-stage detail |
| `error.code`, `error.executed`, `error.attempts`, `failed_stage` | why a run failed, and whether the failed stage may have executed |

Observed on Computer 1 (one run, two loopback processes, not a
benchmark):
- requests were about 12.5 KB and 14.5 KB, mostly the plan JSON;
- responses were 2.2 KB (hidden states) and 8.3 KB (logits);
- round trips were 128 ms (first request, cold plan and stage caches)
  and 26 ms.

## Security

Phase 7 audited every listener: see DEPLOYMENT_LAN.md, "Security
posture". All defaults are loopback now, and TLS / mutual
authentication is described there as future work.

- **No authentication or encryption.** Anyone who can reach the port can
  send requests. Anyone on the network path can read or alter
  activations and plans. TLS and mutual authentication are not
  implemented.
- **Identity checks are consistency checks, not access control.**
  `target_worker_id`, the plan's stage assignment, hostnames and
  `worker_id` in responses stop *misrouted* requests and misconfigured
  coordinators. A malicious client can craft a valid plan assigning any
  stage to any worker. A worker only ever runs the built-in tiny model
  on bounded inputs, which limits the damage to bounded CPU use.
- **Resource safeguards:**
  - bounded messages, tensors, batch and sequence;
  - output size checked before execution;
  - required deadlines;
  - a concurrency cap with `BUSY`;
  - bounded plan and stage caches.
- **Exposure:** loopback by default. Public binds are refused, and
  all-interface binds need an explicit opt-in. Keep it behind the
  firewall rule above.
- **Logs** carry request ids, stage indices, shapes, status codes and
  timings. They never include tensor values, payload bytes or plan
  bodies.

## Known limitations

- **No physical multi-machine run yet.** Everything above ran on
  Computer 1, so cross-machine numerical agreement is untested. The 1e-4
  tolerance is a reasoned allowance, not a measured bound.
- **Only one model:** `builtin:tiny-gpt2-v1`, fp32, a single prefill
  pass, no KV cache or generation, equal-length sequences.
- **Endpoints are given by hand.** The orchestrator doesn't know
  execution endpoints, and there is no automatic placement or scheduling.
- **Hub-and-spoke dataflow** costs two network hops per stage boundary.
  The full plan travels with every request, about 10 KB for the tiny
  model.
- **No authentication, no TLS, no fault-tolerant rescheduling, no
  quantized inference, no GPT-2 execution.**
- **Memory.** Each worker process imports torch and transformers (about
  700 MiB measured in Phase 4). Three worker processes plus the test
  process used several GiB on Computer 1, which runs at about 80% RAM
  use.
