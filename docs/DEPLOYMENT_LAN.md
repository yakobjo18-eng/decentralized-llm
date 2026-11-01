# LAN deployment, preflight and physical test procedure (Phase 7)

This guide sets up the two-computer tiny-model pipeline on a trusted
private LAN, checks it with `preflight`, and records a physical test.

| What | Status |
|---|---|
| Local validation on Computer 1 (orchestrator + 3 worker processes, loopback) | **Done**, 14/14 steps; see [Local validation](#local-validation-on-computer-1) |
| Physical two-computer test (Computer 1 + Computer 2) | **`PENDING`**: Computer 2 is unavailable. Nothing in this repository has run on two computers. |

> **Security.** Every service here (orchestrator REST and gRPC, worker
> Health, StageExecution) is **unauthenticated and unencrypted**. A
> private LAN limits who can reach them; it is not cryptographic
> protection. Use this only on a network where you trust every device.
> Never forward these ports on a router or expose them to the internet.

Related: [REMOTE_EXECUTION.md](REMOTE_EXECUTION.md) (protocol) ·
[NETWORKING.md](NETWORKING.md) · [EXECUTION.md](EXECUTION.md).

## Test classification

Every Python test and every validation step carries exactly one class.
Run a single class with `python -m pytest -m real_multi_process`; each
pytest run prints the counts per class.

| Class | Meaning |
|---|---|
| `REAL_SINGLE_MACHINE` | real code in one OS process on one machine (in-process gRPC over 127.0.0.1 included) |
| `REAL_MULTI_PROCESS` | separate OS processes (orchestrator.exe, worker processes) over real sockets, **one machine** |
| `REAL_MULTI_MACHINE` | separate physical computers. **No test or run is in this class yet.** |
| `SIMULATED` | depends on simulated hardware profiles, monkeypatched hardware/OS sources, synthetic worker records, or stand-in services |

Remote execution results carry `test_classification`, derived from the
hostnames and pids the workers report about themselves (not proof):
- `REAL_MULTI_MACHINE` needs two or more distinct hostnames;
- `REAL_MULTI_PROCESS` means one host with more than one process;
- `REAL_SINGLE_MACHINE` means everything ran in the coordinator's own
  process.

## Topology and ports

```
Computer 1 (C1_IP)                                   Computer 2 (C2_IP)
  orchestrator  gRPC  C1_IP:9090      <── register/heartbeat ──  worker computer-2-b
                REST  127.0.0.1:8080                               Health        127.0.0.1:50100
  worker computer-1-a                                              StageExecution C2_IP:50201
    Health         127.0.0.1:50100
    StageExecution 127.0.0.1:50200
  coordinator (python -m partitioner execute --remote) ── ExecuteStage ──► C2_IP:50201
```

| Listener | Default (code) | LAN setting in this guide | Inbound from | Firewall rule |
|---|---|---|---|---|
| Orchestrator gRPC (WorkerRegistry) | `127.0.0.1:9090` | `C1_IP:9090` | Computer 2 | C1: TCP 9090 from C2_IP |
| Orchestrator REST | `127.0.0.1:8080` | `127.0.0.1:8080` (the coordinator runs on C1) | nobody | none |
| Worker Health | `127.0.0.1:50100` | `127.0.0.1:50100` (the orchestrator never dials workers) | nobody | none |
| Worker StageExecution | off; `127.0.0.1:50200` | C1: `127.0.0.1:50200`; C2: `C2_IP:50201` | the coordinator on C1 | C2: TCP 50201 from C1_IP |

Every listener is **loopback unless configured otherwise** (Phase 7
changed the orchestrator and Health defaults from `0.0.0.0`). Every
listener enforces the same bind policy: loopback and private IPs are
allowed, public IPs and hostnames are refused, and `0.0.0.0` is either
logged as a warning (orchestrator, Health) or needs
`execution_allow_all_interfaces: true` (StageExecution). Only **two**
ports are exposed to the LAN, each to one peer.

## Step 0: same code and dependencies on both computers

```powershell
# Both computers: the same commit.
git clone https://github.com/yakobjo18-eng/decentralized-llm.git; cd decentralized-llm   # or: git pull
git rev-parse HEAD            # must print the same hash on both computers
python --version              # 3.13.x on both (Computer 1: 3.13.3)
python -m pip install -r worker\requirements.txt -r model-partitioner\requirements.txt
```

On Computer 1 only, build the orchestrator (needs Go):

```powershell
cd orchestrator; go build -o orchestrator.exe .\cmd\orchestrator; cd ..
```

torch and transformers are needed on **every** worker: StageExecution
imports them. No model is downloaded; the tiny model's config is built
in.

## Step 1: private IPv4 addresses, subnet, network profile

On each computer:

```powershell
ipconfig | Select-String "IPv4|Subnet"     # e.g. IPv4 Address . . : 192.168.0.199, Subnet Mask . . : 255.255.255.0
Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" } |
  Format-Table InterfaceAlias, IPAddress, PrefixLength
Get-NetConnectionProfile | Format-Table InterfaceAlias, NetworkCategory
```

- **The private IPv4 address** is the one on your Wi-Fi or Ethernet
  adapter, in `192.168.x.x`, `10.x.x.x` or `172.16–31.x.x`.
- **Ignore these:**
  - `169.254.x.x`: APIPA, meaning the adapter got no DHCP lease;
  - `127.x`: loopback.
- **Both computers must be on the same subnet.** With a `/24` mask
  (`255.255.255.0`), the first three numbers must match.
- **Also check:**
  - guest Wi-Fi networks often isolate clients from each other;
  - DHCP can change addresses, so re-check them before every test.
- `python -m partitioner preflight` lists every interface and classifies
  it. It also flags any public IPv4 address on the machine.
- **Network category must be `Private`.** If it shows `Public`, switch it
  only if you trust this network. In an elevated PowerShell:

  ```powershell
  Set-NetConnectionProfile -InterfaceAlias "Wi-Fi" -NetworkCategory Private
  ```

Below, `C1_IP` and `C2_IP` stand for the two addresses you found.

## Step 2: configuration files

**Computer 1, `orchestrator\orchestrator.yaml`:**

```yaml
grpc_address: "C1_IP:9090"        # private LAN IP: Computer 2's worker registers here
rest_address: "127.0.0.1:8080"    # the coordinator and preflight run on Computer 1
heartbeat_timeout_seconds: 15
allow_simulated_workers: false
```

**Computer 1, `worker\worker-a.yaml`:**

```yaml
worker_id: "computer-1-a"
orchestrator_address: "C1_IP:9090"    # the orchestrator listens on C1_IP only, not on 127.0.0.1
listen_host: "127.0.0.1"
listen_port: 50100
execution_enabled: true
execution_listen_host: "127.0.0.1"    # the coordinator is on this machine
execution_listen_port: 50200
```

**Computer 2, `worker\worker-b.yaml`:**

```yaml
worker_id: "computer-2-b"
orchestrator_address: "C1_IP:9090"
listen_host: "127.0.0.1"              # Health: nothing dials it remotely
listen_port: 50100
execution_enabled: true
execution_listen_host: "C2_IP"        # this machine's private LAN IP, never 0.0.0.0
execution_listen_port: 50201
```

The worker ids and execution ports differ on purpose, so every log and
result line is unambiguous.

## Step 3: Windows Firewall, limited to the peer

Run in an **elevated** PowerShell. Each rule allows **one port** from
**one peer IP**, on the Private profile only. `-RemoteAddress
LocalSubnet` is a looser alternative; never use `Any`.

```powershell
# Computer 1: let Computer 2's worker reach the orchestrator's gRPC port.
New-NetFirewallRule -DisplayName "dllm orchestrator gRPC (from C2)" -Direction Inbound -Protocol TCP `
  -LocalPort 9090 -RemoteAddress C2_IP -Action Allow -Profile Private

# Computer 2: let Computer 1's coordinator reach worker B's StageExecution port.
New-NetFirewallRule -DisplayName "dllm StageExecution (from C1)" -Direction Inbound -Protocol TCP `
  -LocalPort 50201 -RemoteAddress C1_IP -Action Allow -Profile Private

# Check (either machine):
Get-NetFirewallRule -DisplayName "dllm*" | Get-NetFirewallPortFilter | Format-Table LocalPort, Protocol
Get-NetFirewallRule -DisplayName "dllm*" | Get-NetFirewallAddressFilter | Format-Table RemoteAddress

# Remove after testing:
Remove-NetFirewallRule -DisplayName "dllm*"
```

- **Program rules can override port rules.** Windows also has
  per-program rules, and a program rule that **blocks** `python.exe` or
  `orchestrator.exe` on the Private profile overrides an allowing port
  rule. Check with:
  ```powershell
  Get-NetFirewallApplicationFilter | Where-Object Program -like "*python*" | Get-NetFirewallRule |
    Format-Table DisplayName, Direction, Action, Profile, Enabled
  ```
- **Finding on Computer 1 (2026-11-01).** Two pre-existing inbound
  **Allow** rules named "Python" exist for
  `C:\Program Files\Python313\python.exe` on the **Public** profile,
  probably from an earlier "allow access" prompt. On a public network,
  they would admit connections to any Python process listening on a
  non-loopback address. They were not changed. Consider removing them,
  or limiting them to Private:
  ```powershell
  Get-NetFirewallRule -DisplayName "Python" | Format-List DisplayName, Profile, Action, Direction
  ```
- A first listen on a LAN address can trigger a Windows Firewall prompt.
  Choose **Private networks only**.

## Step 4: preflight, start, preflight again

Run `preflight` from `model-partitioner\`:
- **Exit code:** 0 means no FAIL; 1 means at least one FAIL, each with a
  hint.
- **Statuses:** `PASS` (tested, fine), `WARN` (tested, needs attention),
  `FAIL` (tested, broken), `SKIP` (**not tested**), `INFO` (facts, no
  verdict).
- **Each check prints what it actually exercised (`tested:`).** An
  endpoint is never called reachable unless a TCP connect and a gRPC
  call to it succeeded.

**Before starting anything, on each computer:**

```powershell
cd model-partitioner
python -m partitioner preflight --worker-config ..\worker\worker-a.yaml     # Computer 1
python -m partitioner preflight --worker-config ..\worker\worker-b.yaml     # Computer 2
```

**Start the services.** Each one runs in its own terminal.

```powershell
# Computer 1, terminal 1
cd orchestrator; .\orchestrator.exe --config orchestrator.yaml
# Computer 1, terminal 2
cd worker; python -m worker.main --config worker-a.yaml
# Computer 2, terminal 1
cd worker; python -m worker.main --config worker-b.yaml
```

- Each worker logs
  `StageExecution (protocol v1, UNAUTHENTICATED) listening on ...` and
  `registered with orchestrator`.
- Its inference backend probe then finishes about 5–60 s later, and the
  worker re-registers. Until then, preflight shows a **WARN**
  ("backend probe is still running"), and planning refuses that worker.

**From Computer 1, check reachability, registration and protocol:**

```powershell
Test-NetConnection -ComputerName C2_IP -Port 50201       # quick firewall check: TcpTestSucceeded : True
python -m partitioner preflight --orchestrator http://127.0.0.1:8080 `
  --expect-worker computer-1-a --expect-worker computer-2-b `
  --endpoint computer-1-a=127.0.0.1:50200 --endpoint computer-2-b=C2_IP:50201
```

### How preflight reads connection failures

| Category | Meaning | Typical fix |
|---|---|---|
| `dns` | the name does not resolve | use the IPv4 address instead of a hostname |
| `refused` | the host answered, but nothing listens on that port | start the worker; check the port, `execution_enabled` and the bind IP (a service on 127.0.0.1 refuses LAN clients) |
| `timeout` | no answer at all | usually the Windows Firewall on the target (Step 3), a wrong IP, or the machine is asleep. On a **loopback** target it usually means nothing is listening (see the note below). |
| `unreachable` | no route | different network or subnet; client isolation |
| gRPC error after TCP success | something else owns the port (e.g. the Health port), or it is not a StageExecution server | point `--endpoint` at `execution_listen_port` |
| protocol `FAIL` | worker and coordinator speak different protocol versions | run the same commit everywhere |
| protocol `WARN` "predates GetInfo" | the worker runs the Phase 6 code | update it; preflight used a rejected-version probe that executes nothing |
| identity `FAIL` | the endpoint answers as another `worker_id` | wrong IP or port, or a `worker_id` typo |

On Windows, a connection to a closed port is reported as refused only
after about 2 s of retries (measured on Computer 1: 2.0 s, loopback and
own LAN IP alike). With a shorter `--timeout`, a closed port looks like a
timeout. Keep the default 3 s.

## Step 5: plan from the real profiles, execute, record

```powershell
cd model-partitioner
python -m partitioner plan --model builtin:tiny-gpt2-v1 --orchestrator http://127.0.0.1:8080 `
  --require-all-workers --batch 2 --out lan-plan.json
python -m partitioner preflight --orchestrator http://127.0.0.1:8080 --plan lan-plan.json --require-multi-host `
  --endpoint computer-1-a=127.0.0.1:50200 --endpoint computer-2-b=C2_IP:50201 --out lan-preflight.json
python -m partitioner execute lan-plan.json --remote --tokens "1,2,3,4,5,6,7,8;9,10,11,12,13,14,15,16" `
  --endpoint computer-1-a=127.0.0.1:50200 --endpoint computer-2-b=C2_IP:50201 --out lan-result.json
python ..\scripts\lan_result_summary.py lan-result.json --label "C1+C2 first run"
```

**If planning reports INFEASIBLE, do not fake capacity.** Every stage
needs the measured ~700 MiB runtime overhead, and Computer 1 has had
little free RAM in every phase (in the 2026-11-01 validation, the
largest effective capacity was 52–82 MiB under the default Phase 3
margins). Honest options:
- close other programs on the affected machine;
- explicitly lower the orchestrator's `capability:` margins
  (`ram_safety_margin_percent`, `ram_safety_margin_min_gib` in
  `orchestrator.yaml`), and record the values you used in the results
  table;
- lower the planner's `--extra-safety`.

Whatever you change, write it in the "Deviations" row.

## Physical test checklist (Computer 1 + Computer 2)

Classification target: **REAL_MULTI_MACHINE**. Tick each item as you go,
and paste the evidence (command output) into the results file.

1. [ ] Same commit on both computers (`git rev-parse HEAD`); Python 3.13 on both.
2. [ ] Private IPv4 addresses, same subnet, profile `Private` on both (Step 1).
3. [ ] Configs as in Step 2; `preflight --worker-config` shows no FAIL on each computer.
4. [ ] Firewall rules from Step 3, limited to the peer IP.
5. [ ] Start the orchestrator on C1, worker A on C1 and worker B on C2.
6. [ ] Both workers are **registered, online, REAL**, with real hardware profiles:
   `curl.exe http://127.0.0.1:8080/api/workers`. Each `hostname` is the
   real machine's name, and `profile_source` is `REAL`.
7. [ ] `Test-NetConnection C2_IP -Port 50201` succeeds from C1.
8. [ ] Preflight with `--orchestrator`, both `--endpoint`s and `--expect-worker`s shows no FAIL.
9. [ ] Plan generated **from the orchestrator (real profiles only)** with
   `--require-all-workers`, and preflight `--plan ... --require-multi-host`
   shows no FAIL. The plan has one stage per computer.
10. [ ] `execute --remote` succeeded, and the reference check passed
    (`max_abs_diff ≤ 1e-4`).
11. [ ] `lan_result_summary.py` shows **two distinct reported
    hostnames** and `test_classification: REAL_MULTI_MACHINE`.
12. [ ] **Disconnect and restart:**
    - stop worker B (Ctrl+C) and confirm the orchestrator drops it;
    - run `execute` again: it is expected to **fail** at B's stage with
      `RETRIES_EXHAUSTED`, `executed: false`;
    - restart B, wait for "registered", and run `execute` again: it is
      expected to succeed;
    - then disable C2's network adapter for about 30 s and re-enable it;
      confirm B goes offline, comes back by itself, and that `execute`
      succeeds;
    - restart the orchestrator and confirm both workers re-register.
13. [ ] Remove the firewall rules (`Remove-NetFirewallRule -DisplayName "dllm*"`).

### Results (fill in; leave `PENDING` until it has actually run)

| Field | First run | After disconnect/restart |
|---|---|---|
| Date, operator | `PENDING` | `PENDING` |
| Commit (`git rev-parse HEAD`) | `PENDING` | `PENDING` |
| Computer 1 hostname / IP | `PENDING` | `PENDING` |
| Computer 2 hostname / IP | `PENDING` | `PENDING` |
| Worker ids (A / B) | `PENDING` | `PENDING` |
| Model id | `PENDING` (expected `builtin:tiny-gpt2-v1`) | `PENDING` |
| Plan id (`inputs_fingerprint`) | `PENDING` | `PENDING` |
| Stage → worker → reported host | `PENDING` | `PENDING` |
| Output shape | `PENDING` (expected `[2, 8, 128]`) | `PENDING` |
| Next token ids | `PENDING` (expected `[60, 96]`, the same as on Computer 1) | `PENDING` |
| Reference max abs diff / tolerance | `PENDING` / `1e-4` | `PENDING` / `1e-4` |
| Elapsed (coordinator, single run, not a benchmark) | `PENDING` | `PENDING` |
| `test_classification` | `PENDING` (target `REAL_MULTI_MACHINE`) | `PENDING` |
| Deviations from this guide (margins, ports, …) | `PENDING` | `PENDING` |
| Result | **`PENDING`** | **`PENDING`** |

## Local validation on Computer 1

`scripts/phase7_local_validation.py` runs the operational checks with
the real `orchestrator.exe` and three `python -m worker.main` processes,
all on 127.0.0.1. Every step is `REAL_MULTI_PROCESS`, except S01, which
is `REAL_SINGLE_MACHINE`. Re-run it with:

```powershell
python scripts\phase7_local_validation.py --out docs\validation\phase7-local-<date>
```

Recorded run: [validation/phase7-local-2026-11-01-run2/report.md](validation/phase7-local-2026-11-01-run2/report.md),
**14/14 steps passed**. Its `report.json` and `logs/` hold the evidence.
The first recorded run
([run1](validation/phase7-local-2026-11-01-run1/report.md)) passed 13/14:
step S09 failed because of a **bug in the validation script**. It assumed
`val-a` owned stage 0, but the planner put `val-b` first in that run, and
the worker answered both swapped requests correctly. The script now
reads stage ownership from the plan.

Run 2 in brief (single-run timings on a loaded laptop, not benchmarks):

| Step | Outcome |
|---|---|
| S01 preflight on configs | no FAIL |
| S02 start orchestrator + 3 workers | all online and REAL; workers listening about 10 s after start (torch + transformers import) |
| S03 exposure audit | 8 listening sockets, **all on 127.0.0.1** (psutil, actual bound addresses) |
| S04 plan from live registrations | **INFEASIBLE**: largest effective capacity 54 MiB vs. 700 MiB needed. Pipelines used the same REAL records with a **synthetic** 3 GiB capacity, labelled as such |
| S05 preflight, live | no FAIL; 1 WARN: the plan's synthetic capacity exceeds the workers' live usable RAM, correctly flagged |
| S06 / S07 2- and 3-worker pipelines | succeeded; logits `[2, 8, 128]` / `[1, 8, 128]`; reference diff 0.0; 2 and 3 processes on 1 host |
| S08 resource limit | `LIMIT_EXCEEDED` at `val-c` (`max_batch 1`), not executed |
| S09 malformed requests | `PROTOCOL_VERSION`, `MALFORMED_REQUEST`, `INVALID_TENSOR`, `UNASSIGNED_STAGE`, `DEADLINE`; the worker kept serving |
| S10 20 ms deadline | rejected `DEADLINE` (< 0.05 s minimum), not retried |
| S11 graceful stop and restart | exit code 0, deregistered; while it was down, `RETRIES_EXHAUSTED` (`executed: false`); back in 8.1 s with a new pid |
| S12 hard kill | marked offline after 2.8 s (timeout 3 s); back 0.5 s after restart |
| S13 orchestrator restart | pipeline **succeeded while the orchestrator was down**; all workers re-registered 0.8 s after it came back |
| S15 clean shutdown | all exit code 0, `worker shutdown complete`, deregistered |

Not run: the optional `--lan-ip` step. Binding a worker to Computer 1's
LAN IP (192.168.0.199 at the time) could open a Windows Firewall prompt
on the desktop, and traffic to one's own IP never leaves the machine
anyway.

## Security posture (audit, 2026-11-01)

| Listener | Code default | Policy enforced at startup | Authentication | Encryption |
|---|---|---|---|---|
| Orchestrator REST | 127.0.0.1:8080 | Go `config.CheckBindAddress`: public IP or hostname → refuse to start; 0.0.0.0 → warning | none | none |
| Orchestrator gRPC | 127.0.0.1:9090 | same | none | none |
| Worker Health | 127.0.0.1:50100 | `worker/config/bind_policy.py`: public IP or hostname → refuse; 0.0.0.0 → warning | none | none |
| StageExecution | off; 127.0.0.1:50200 | same, but 0.0.0.0 also needs `execution_allow_all_interfaces: true` | none | none |

- **Verified.**
  - The bind policies are covered by unit tests, Go and Python.
  - On Computer 1 the actual bound addresses during validation were all
    127.0.0.1 (S03).
  - `preflight` warns when a machine has a public IPv4 address. Computer
    1 has none: Wi-Fi 192.168.0.199 is private, and two 169.254 APIPA
    adapters are listed.
- **What is not protected:**
  - anyone who can reach a LAN-bound port can register workers (the
    orchestrator) or run stages (StageExecution);
  - traffic can be read and altered on the network path;
  - worker ids, hostnames and plans are self-asserted.

  The assignment and identity checks prevent mistakes, not attacks.
- **Future work, not implemented:**
  - gRPC with TLS (`grpc.ssl_server_credentials` /
    `ssl_channel_credentials`, Go `credentials.NewTLS`), with mutual TLS
    so workers and the coordinator authenticate each other with
    per-machine certificates from a private CA;
  - or, at minimum, server-authenticated TLS plus a shared bearer token
    checked in an interceptor.

  Either would use the standard gRPC/TLS libraries. **No custom
  cryptography.** None of this exists yet.
