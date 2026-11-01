# LAN deployment and two-machine test procedure (Phases 7–8)

This guide runs the tiny-model pipeline with the orchestrator and one
worker on **Computer 1** and a second worker on **Computer 2**, on a
trusted private LAN. It covers how to set it up, check it, run it,
record the result and shut it down.

| What | Status |
|---|---|
| Tooling: config generators, `preflight` (incl. firewall, ports, peer evidence), `scripts/phase8_lan.py run` | **Ready**, tested locally |
| Loopback rehearsal of the full test on Computer 1 | **Done**, `REAL_MULTI_PROCESS`: [report](validation/phase8-loopback-rehearsal-2026-11-01/report.md) |
| Physical two-computer test (Computer 1 + Computer 2) | **`PENDING`**: no second computer has taken part yet. Nothing here is `REAL_MULTI_MACHINE`. |

> **Security.** Every service (orchestrator REST and gRPC, worker
> Health, StageExecution) is **unauthenticated and unencrypted**. A
> private LAN limits who can reach them; it is not cryptographic
> protection. Use only a network where you trust every device. Never
> forward these ports on a router or expose them to the internet.

Related: [REMOTE_EXECUTION.md](REMOTE_EXECUTION.md) (protocol) ·
[NETWORKING.md](NETWORKING.md) · [EXECUTION.md](EXECUTION.md) ·
[validation/README.md](validation/README.md).

## Test classification

| Class | Meaning |
|---|---|
| `REAL_SINGLE_MACHINE` | real code in one OS process on one machine |
| `REAL_MULTI_PROCESS` | separate OS processes over real sockets, **one machine** |
| `REAL_MULTI_MACHINE` | separate physical computers. **Nothing is in this class yet.** |
| `SIMULATED` | depends on simulated hardware, monkeypatched OS sources, synthetic records or stand-in services |

`scripts/phase8_lan.py run` reports `REAL_MULTI_MACHINE` **only if both**
of these hold:
1. the stage workers report **two or more distinct hostnames**
   (self-asserted);
2. the orchestrator saw their registrations arrive from **two or more
   distinct non-loopback IPs**. This is the TCP peer address, recorded
   since Phase 8 as `observed_peer_ip`.

Either one alone is not enough: hostnames are just strings, and one
multi-homed machine can have two IPs. If the criterion is not met, the
run **fails** that step. The only exception is `--allow-single-host`,
which marks the run as a rehearsal and keeps the honest class.

## Topology and ports

```
Computer 1 (C1_IP)                                         Computer 2 (C2_IP)
  orchestrator  gRPC  C1_IP:9090    <── Register/Heartbeat ──  worker computer-2-b
                REST  127.0.0.1:8080 (or another free port)      Health         127.0.0.1:50100
  worker computer-1-a                                            StageExecution C2_IP:50201
    Health         127.0.0.1:50100
    StageExecution 127.0.0.1:50200
  coordinator: scripts/phase8_lan.py run ──── ExecuteStage ────►  C2_IP:50201
```

| Listener | Code default | Setting in this guide | Must accept connections from | Firewall rule |
|---|---|---|---|---|
| Orchestrator gRPC | `127.0.0.1:9090` | `C1_IP:9090` | Computer 2 | **C1:** TCP 9090, remote `C2_IP` only |
| Orchestrator REST | `127.0.0.1:8080` | `127.0.0.1:<free port>` | nobody (the coordinator is on C1) | none |
| Worker Health (both) | `127.0.0.1:50100` | `127.0.0.1:50100` | nobody (the orchestrator never dials workers) | none |
| StageExecution, worker A (C1) | off | `127.0.0.1:50200` | the coordinator, same machine | none |
| StageExecution, worker B (C2) | off | `C2_IP:50201` | the coordinator on C1 | **C2:** TCP 50201, remote `C1_IP` only |

- **Only two ports face the LAN**, each from one peer. Every listener
  refuses public IPs and hostnames at startup; `0.0.0.0` either logs a
  warning or needs an explicit opt-in.
- **Port conflicts are real.** On Computer 1 in Phase 8, **port 8080 was
  held by an unrelated program** (a Java process on `0.0.0.0:8080`). If
  preflight reports `held by an unrelated program`, choose another REST
  port, e.g. `--rest-port 8081`, and use it in `--orchestrator-url`.
- **Windows can reserve port ranges** (`WinError 10013`). Check with:
  ```powershell
  netsh int ipv4 show excludedportrange protocol=tcp
  ```
  Computer 1 reserved 50000–50059, so the defaults (50100, 50200, 50201)
  are outside it.

## Step 1: same code and dependencies on both computers

```powershell
git clone https://github.com/yakobjo18-eng/decentralized-llm.git; cd decentralized-llm   # or: git pull
git rev-parse HEAD                 # MUST be identical on both computers
python --version                   # 3.13.x on both
python -m pip install -r worker\requirements.txt -r model-partitioner\requirements.txt
cd orchestrator; go build -o orchestrator.exe .\cmd\orchestrator; cd ..     # Computer 1 only (needs Go)
```

Workers need torch and transformers; nothing is downloaded. A
different commit on the two machines shows up in preflight as a
protocol `FAIL` or a "predates GetInfo" `WARN`.

## Step 2: find the private IPv4 addresses

On **each** computer:

```powershell
Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" } |
  Format-Table InterfaceAlias, IPAddress, PrefixLength
Get-NetConnectionProfile | Format-Table InterfaceAlias, NetworkCategory
cd model-partitioner; python -m partitioner preflight --skip-runtime    # also lists and classifies every interface
```

- **Computer 1's LAN IP (`C1_IP`)** is the private address on the adapter
  connected to the shared network (Wi-Fi or Ethernet). It is a
  `192.168.x.x`, `10.x.x.x` or `172.16–31.x.x` address.
- **Ignore** `169.254.x.x` (APIPA: no DHCP lease), `127.x`, and adapters
  belonging to VPNs or virtual switches.
- **Both computers must be on the same subnet.** With a `/24`
  (`PrefixLength 24`), the first three numbers must match. Guest Wi-Fi
  often isolates clients.
- **NetworkCategory must be `Private`** on both. If it isn't, and you
  trust the network, run this in an elevated PowerShell:
  ```powershell
  Set-NetConnectionProfile -InterfaceAlias "<adapter>" -NetworkCategory Private
  ```
- **Write the values down** (`C1_IP`, `C2_IP`). DHCP can change them, so
  re-check before every session.

## Step 3: generate the configs (all values from the command line)

`scripts/phase8_lan.py` writes validated YAML. It refuses public IPs,
`0.0.0.0`, hostnames, duplicate ports, and overwriting an existing file
(unless you pass `--force`). Run from the repository root:

```powershell
# Computer 1: orchestrator (gRPC on the LAN, REST local) and worker A (StageExecution local: the coordinator is here).
python scripts\phase8_lan.py orchestrator-config --lan-ip C1_IP --grpc-port 9090 --rest-port 8080 --out orchestrator\orchestrator.yaml
python scripts\phase8_lan.py worker-config --worker-id computer-1-a --orchestrator C1_IP:9090 `
  --execution-host 127.0.0.1 --execution-port 50200 --allow-loopback --out worker\worker-a.yaml

# Computer 2: worker B (StageExecution on its LAN IP).
python scripts\phase8_lan.py worker-config --worker-id computer-2-b --orchestrator C1_IP:9090 `
  --execution-host C2_IP --execution-port 50201 --out worker\worker-b.yaml
```

- **Worker A's `orchestrator_address` must be `C1_IP:9090`, not
  `127.0.0.1`.** The orchestrator listens on `C1_IP` only.
- **Worker ids must be unique.** Since Phase 8 the orchestrator also
  compares the connection's source IP. If Computer 2 reuses Computer 1's
  id while it is online, registration is rejected with
  `worker id is already registered and online from a different address
  ... (from peer …)`. Without that check, it would have silently taken
  over the id.
- If live planning turns out infeasible (Step 7), you can regenerate the
  orchestrator config with `--ram-margin-percent` /
  `--ram-margin-min-gib`. The file then carries a `# DEVIATION` comment;
  record it in the results.

You can also write the YAML by hand. These are the generator's outputs:
```yaml
# orchestrator\orchestrator.yaml (C1)
grpc_address: "C1_IP:9090"
rest_address: "127.0.0.1:8080"
heartbeat_timeout_seconds: 15
allow_simulated_workers: false
# worker\worker-b.yaml (C2)
worker_id: "computer-2-b"
orchestrator_address: "C1_IP:9090"
listen_host: "127.0.0.1"
listen_port: 50100
heartbeat_interval_seconds: 5
execution_enabled: true
execution_listen_host: "C2_IP"
execution_listen_port: 50201
simulated_profile: ""
```

## Step 4: Windows Firewall, least privilege

Run in an **elevated** PowerShell. Each rule allows one TCP port from
one peer IP, on the Private profile only:

```powershell
# Computer 1: Computer 2's worker -> orchestrator gRPC
New-NetFirewallRule -DisplayName "dllm orchestrator gRPC (from C2)" -Direction Inbound -Protocol TCP `
  -LocalPort 9090 -RemoteAddress C2_IP -Action Allow -Profile Private
# Computer 2: Computer 1's coordinator -> worker B StageExecution
New-NetFirewallRule -DisplayName "dllm StageExecution (from C1)" -Direction Inbound -Protocol TCP `
  -LocalPort 50201 -RemoteAddress C1_IP -Action Allow -Profile Private
# Remove when done (both machines):
Remove-NetFirewallRule -DisplayName "dllm*"
```

- **No other inbound rules are needed.** Outbound traffic is allowed by
  default.
- **Block rules win over allow rules.** If you ever clicked **Cancel** on
  a Windows Firewall prompt for `python.exe` or `orchestrator.exe`,
  Windows created a Block rule that silently overrides the rules above.
  Preflight detects this (next step).
- **Broad rules break least privilege.** An Allow rule with
  `RemoteAddress Any` (for example, created by an "allow access" prompt)
  opens the port to the whole network; preflight reports it as `WARN`.
- **Finding on Computer 1 (Phase 7/8).** Pre-existing inbound Allow
  rules admit `python.exe` ("Python", 2 rules) and `orchestrator.exe` (2
  rules) **on any port on the Public profile**. They don't apply on a
  Private network, but would on a public one. They were not modified;
  review them with:
  ```powershell
  Get-NetFirewallRule -DisplayName "Python","orchestrator.exe" | Format-Table DisplayName, Profile, Direction, Action, Enabled
  ```
- **How preflight reads the firewall.** It uses
  `netsh advfirewall ... verbose`, which is read-only and took about
  0.4 s on Computer 1; the PowerShell cmdlets took 65 s. It also accounts
  for the active profile, Block rules, port lists and ranges, program
  paths, and the rule's local IP. It is a prediction: the real test is a
  connection from the peer. On a non-English Windows the output is
  localized and the check reports `SKIP` rather than guessing.

## Step 5: preflight, before starting anything

Run from `model-partitioner\`:

```powershell
# Computer 1
python -m partitioner preflight --orchestrator-config ..\orchestrator\orchestrator.yaml --worker-config ..\worker\worker-a.yaml --peer-ip C2_IP
# Computer 2
python -m partitioner preflight --worker-config ..\worker\worker-b.yaml --peer-ip C1_IP
```

- **Exit code:** 0 means no FAIL; 1 means at least one FAIL, each with a
  hint.
- **Statuses:** `PASS` (tested, fine), `WARN` (tested, needs attention),
  `FAIL` (tested, broken), `SKIP` (**not tested**), `INFO` (facts, no
  verdict). Every line says what it actually exercised (`tested:`).

Expected **before** the firewall rules exist. This is the format from a
Computer 1 run, with addresses replaced by placeholders:

```
[PASS] grpc_address: C1_IP:9090 (private)
[PASS] grpc_address port 9090: free
[FAIL] grpc_address firewall: no enabled inbound Allow rule matches TCP 9090 on the ['Private'] profile: connections
       from the peer will be dropped (they show up there as a timeout)
       hint: In an elevated PowerShell: New-NetFirewallRule -DisplayName "dllm grpc_address 9090" -Direction Inbound
             -Protocol TCP -LocalPort 9090 -RemoteAddress C2_IP -Action Allow -Profile Private
[WARN] grpc_address public-profile rules: 2 enabled inbound Allow rule(s) ['orchestrator.exe'] admit this program on ANY port ...
[FAIL] rest_address port 8080: held by an unrelated program: java.exe (pid ...)
```

- **After Step 4:** the `firewall` lines turn `PASS`, as "allowed only
  from specific remote addresses".
- **A port held by another program** stays `FAIL` until you pick a free
  port.
- **A port held by our own program** shows as
  `held by orchestrator.exe ... probably this service already running`
  (`INFO`), or `held by python.exe` (`WARN`).

## Step 6: startup order

Start each service in its own terminal, in this order:

| # | Where | Command | Wait for this log line |
|---|---|---|---|
| 1 | C1 | `cd orchestrator; .\orchestrator.exe --config orchestrator.yaml` | `gRPC WorkerRegistry service listening on C1_IP:9090` and `REST API listening on 127.0.0.1:8080` |
| 2 | C1 | `cd worker; python -m worker.main --config worker-a.yaml` | `StageExecution (protocol v1, UNAUTHENTICATED) listening on 127.0.0.1:50200`, then `registered with orchestrator` |
| 3 | C2 | `cd model-partitioner; python -m partitioner preflight --worker-config ..\worker\worker-b.yaml --probe-orchestrator --peer-ip C1_IP` | `[PASS] orchestrator_address reachable: TCP connect to C1_IP:9090 succeeded`. A `timeout` here means C1's firewall (Step 4); `refused` means the orchestrator isn't running or is on another IP/port. |
| 4 | C2 | `cd worker; python -m worker.main --config worker-b.yaml` | `StageExecution ... listening on C2_IP:50201`, then `registered with orchestrator` |

Then, from Computer 1:

```powershell
Test-NetConnection -ComputerName C2_IP -Port 50201        # TcpTestSucceeded : True
curl.exe http://127.0.0.1:8080/api/workers                # both workers: online, REAL, observed_peer_ip = C1_IP / C2_IP
cd model-partitioner
python -m partitioner preflight --orchestrator http://127.0.0.1:8080 --expect-worker computer-1-a --expect-worker computer-2-b `
  --endpoint computer-1-a=127.0.0.1:50200 --endpoint computer-2-b=C2_IP:50201
```

- **Backend probe.** Each worker's inference backend probe finishes
  about 5–60 s after it starts. Until then preflight shows `WARN`
  ("backend probe is still running"), and planning refuses that worker.
- **Expected shape of a passing live preflight.** This is a real excerpt
  from the Phase 8 loopback rehearsal; on the LAN, worker B's endpoint
  and "connected from" show C2_IP:

```
[INFO] registered workers: rehearsal-a (online, REAL, host YakobsPC, reported addr 127.0.0.1:56786, connected from 127.0.0.1), ...
[PASS] worker rehearsal-a: online, REAL, host YakobsPC, heartbeat 0.0s ago
[PASS] TCP reachable: TCP connect to 127.0.0.1:56785 succeeded
[PASS] protocol: worker speaks [1], coordinator speaks 1
[PASS] identity: answers as 'rehearsal-a' on host 'YakobsPC' (pid 21336); self-reported, unauthenticated
SUMMARY: 25 PASS, 0 WARN, 0 FAIL, 1 SKIP, 8 INFO -- no failures
```

## Step 7: run the test (Computer 1)

```powershell
python scripts\phase8_lan.py run --orchestrator-url http://127.0.0.1:8080 `
  --endpoint computer-1-a=127.0.0.1:50200 --endpoint computer-2-b=C2_IP:50201 `
  --repeat 3 --interactive --out docs\validation\phase8-lan-<date>
```

| Step | What it does | Pass condition |
|---|---|---|
| P1 | records coordinator host, IPs, commit (and whether there are uncommitted changes) | informational |
| P2 | waits until both workers are registered, online, and done probing | within `--probe-timeout` (240 s) |
| P3 | live preflight: orchestrator, registrations, endpoints, protocol, identity | no FAIL |
| P4 | plans the tiny model **from live registrations only** (`--require-all-workers` semantics) | feasible; one stage per worker. **There is no synthetic fallback.** If infeasible, the run stops with the planner's reasons. |
| P5 | preflight with the plan: assignment, limits, `--require-multi-host` | no FAIL |
| P6 | executes `--repeat` times and compares with the local reference | all succeed, max abs diff ≤ 1e-4 |
| P7 | evidence-based classification (see above) | `REAL_MULTI_MACHINE` |
| P8 | with `--interactive`, prompts you to stop worker B, then checks that execution fails safely (`executed: false`), that B can be restarted and recovers, and that an orchestrator restart is recovered from | each recovers |

- **Output:** `report.md`, `report.json`, `preflight-*.txt`, `plan.json`
  and `result-*.json` in `--out`. The report includes a per-worker
  evidence table (reported host, pid, endpoint, the IP the orchestrator
  saw it register from).
- **Exit code:** 0 only if every required step passed.
- **The run keeps your real IPs in its files**, because they are the
  evidence. Decide yourself whether to commit them. Rehearsal reports
  redact them.

Manual equivalent (the same building blocks):

```powershell
cd model-partitioner
python -m partitioner plan --model builtin:tiny-gpt2-v1 --orchestrator http://127.0.0.1:8080 --require-all-workers --batch 2 --out lan-plan.json
python -m partitioner preflight --orchestrator http://127.0.0.1:8080 --plan lan-plan.json --require-multi-host `
  --endpoint computer-1-a=127.0.0.1:50200 --endpoint computer-2-b=C2_IP:50201
python -m partitioner execute lan-plan.json --remote --tokens "1,2,3,4,5,6,7,8;9,10,11,12,13,14,15,16" `
  --endpoint computer-1-a=127.0.0.1:50200 --endpoint computer-2-b=C2_IP:50201 --out lan-result.json
python ..\scripts\lan_result_summary.py lan-result.json --label "C1+C2"
```

**If P4 reports INFEASIBLE.** Every stage carries the measured ~700 MiB
runtime overhead, and Computer 1's effective capacity was 52–82 MiB in
the Phase 7 runs under the default margins. Close programs, or
regenerate the orchestrator config with lower `--ram-margin-*` values
(a recorded deviation) and restart the orchestrator. Never edit plans by
hand: edited plans are rejected by their fingerprint.

## Step 8: shutdown

1. Let the coordinator finish; `run` stops on its own.
2. **Computer 2:** press Ctrl+C in worker B's terminal. It logs
   `worker shutdown complete` and deregisters, and the orchestrator
   lists it no more.
3. **Computer 1:** press Ctrl+C in worker A's terminal, then in the
   orchestrator's terminal.
4. **Both machines:** remove the firewall rules and confirm nothing is
   still listening:
   ```powershell
   Remove-NetFirewallRule -DisplayName "dllm*"
   netstat -ano | findstr ":9090 :50200 :50201"
   ```
5. Optionally return the network profile to what it was.

Ctrl+C is the graceful path: the worker deregisters immediately. Closing
the window or killing the process skips deregistration; the
orchestrator then marks the worker offline after
`heartbeat_timeout_seconds` (15 s by default).

## Troubleshooting

| Symptom | Likely cause | Check / fix |
|---|---|---|
| C2 preflight `orchestrator_address reachable`: **timeout** | C1 firewall drops 9090, wrong `C1_IP`, different subnet | Step 4 rule on C1; `ipconfig` on both; `Test-NetConnection C1_IP -Port 9090` from C2 |
| …: **refused** | orchestrator not running, or listening on another IP/port (e.g. still `127.0.0.1`) | C1 log line `gRPC WorkerRegistry service listening on C1_IP:9090` |
| …: **dns** | a hostname was used | use the IPv4 address |
| Worker B never appears in `/api/workers` | as above, or its `orchestrator_address` is wrong | B's log: `orchestrator unreachable at … retrying` |
| Registration rejected: `worker id is already registered and online from a different address … (from peer …)` | two workers share a `worker_id` | give each machine its own id. A stale registration expires after the heartbeat timeout. |
| Registration rejected: `does not accept SIMULATED hardware profiles` | `simulated_profile` is set | empty it for a real test |
| Preflight `worker …: backend probe is still running` | normal for 5–60 s after start | wait; planning refuses the worker until then |
| Endpoint **timeout** from C1 | C2 firewall (50201), a Block rule, wrong `C2_IP`, C2 asleep | preflight on C2 (`… firewall` line); `Test-NetConnection C2_IP -Port 50201` |
| Endpoint **refused** | worker B not running, `execution_enabled` false, or bound to 127.0.0.1 | B's log `StageExecution … listening on C2_IP:50201` |
| Endpoint `TCP is open but the gRPC call failed` | `--endpoint` points at the Health port or another program | use `execution_listen_port` |
| `identity`: answers as another worker | wrong IP or port, or an id typo | fix `--endpoint` |
| `protocol`: worker speaks […], coordinator speaks … / "predates GetInfo" | different commits on the two machines | `git rev-parse HEAD` on both; pull and restart |
| Port line: `held by an unrelated program` | another program uses the port (8080 on Computer 1 in Phase 8) | choose another port everywhere it appears |
| `WinError 10013` binding a port | Windows excluded port range, or an exclusive bind by another program | `netsh int ipv4 show excludedportrange protocol=tcp` |
| Plan stage: `endpoint … is loopback but the stage is planned for host …` | used 127.0.0.1 for the remote worker | use `C2_IP:50201` |
| Plan stage: `registered from X but endpoint is Y` (note) | multi-homed machine or wrong IP | make sure `--endpoint` is the address of B's LAN adapter |
| Execution `STAGE_MISMATCH … planned for host` | plan made while a hostname differed, or for another machine | re-plan from the live orchestrator (`run` does) |
| Execution `RETRIES_EXHAUSTED`, `executed: false` | a worker became unreachable; nothing ran there | restart it; re-run. This is the expected result in P8. |
| Execution `AMBIGUOUS_FAILURE` (deadline) | slow link or machine | raise `--stage-timeout`; it is never retried automatically |
| P4 INFEASIBLE | not enough usable RAM under the margins | see Step 7 |
| P7 FAIL `multi-machine criterion NOT met` | both workers on one machine (or one registered via loopback) | that run was not a two-machine test; check where worker B really runs |
| Things worked, then stopped | DHCP changed an IP, Wi-Fi changed network, machine slept | re-run Step 2 and preflight |

## Physical test checklist (Computer 1 + Computer 2)

Classification target: **`REAL_MULTI_MACHINE`**.

1. [ ] Step 1: same commit on both (`git rev-parse HEAD`), Python 3.13, dependencies installed, orchestrator built on C1.
2. [ ] Step 2: `C1_IP` and `C2_IP` written down; same subnet; profile `Private` on both.
3. [ ] Step 3: configs generated (or written) with unique worker ids.
4. [ ] Step 4: two firewall rules, each limited to the peer IP.
5. [ ] Step 5: preflight on each machine with no FAIL (firewall lines PASS, ports free).
6. [ ] Step 6: services started in order; C2 `--probe-orchestrator` PASS; both workers online and REAL; `observed_peer_ip` shows `C1_IP` for A and `C2_IP` for B.
7. [ ] Step 7: `phase8_lan.py run … --interactive` → **RESULT PASS, classification `REAL_MULTI_MACHINE`**, P8 recovered.
8. [ ] Fill in the table below from `report.md`, and keep the `--out` directory.
9. [ ] Step 8: shutdown, rules removed, ports closed.

### Results (leave `PENDING` until it has actually run)

| Field | First run | After disconnect/restart (P8) |
|---|---|---|
| Date, operator | `PENDING` | `PENDING` |
| Commit on C1 / C2 | `PENDING` | `PENDING` |
| C1 hostname / IP; C2 hostname / IP | `PENDING` | `PENDING` |
| Worker ids → reported host → orchestrator-observed peer IP | `PENDING` | `PENDING` |
| Model id / plan id | `PENDING` (expected `builtin:tiny-gpt2-v1`) | `PENDING` |
| Plan source | `PENDING` (must be "generated from LIVE orchestrator registrations") | `PENDING` |
| Output shape / next token ids | `PENDING` (expected `[2, 8, 128]` / `[60, 96]`) | `PENDING` |
| Reference max abs diff / tolerance | `PENDING` / `1e-4` | `PENDING` / `1e-4` |
| Elapsed per run (single runs, not a benchmark) | `PENDING` | `PENDING` |
| `phase8_lan.py` classification | `PENDING` (target `REAL_MULTI_MACHINE`) | `PENDING` |
| Deviations (margins, ports, …) | `PENDING` | `PENDING` |
| Result | **`PENDING`** | **`PENDING`** |

## What has been validated locally

- **Phase 7:** [local validation](validation/phase7-local-2026-11-01-run2/report.md),
  14/14, `REAL_MULTI_PROCESS`. Real orchestrator + 3 `worker.main`
  processes: pipelines, limits, malformed requests, timeouts, worker
  stop/restart, hard kill, orchestrator restart, clean shutdown, all
  listeners on 127.0.0.1.
- **Phase 8:** [loopback rehearsal](validation/phase8-loopback-rehearsal-2026-11-01/report.md)
  of `phase8_lan.py run`, P1–P8 all PASS, **classification
  `REAL_MULTI_PROCESS`**. Details:
  - 3 executions matched the reference exactly (diff 0.0);
  - with the worker down, execution failed with `executed=False`;
  - the worker was back after 3.6 s; workers re-registered 1.1 s after
    the orchestrator restart;
  - the orchestrator recorded both registrations as coming from
    127.0.0.1, so P7 classified honestly.

  The plan used the same REAL records with **synthetic capacity**,
  because live planning was infeasible on Computer 1. That fallback
  exists only in `rehearse`, never in `run`.
- **Phase 8 strict rehearsal:** [strict demonstration](validation/phase8-loopback-strict-2026-11-01/report.md)
  is the same rehearsal without `--allow-single-host`. **RESULT FAIL by
  design:** P5 `hosts spanned: 1` and P7 `multi-machine criterion NOT
  met`, showing that one machine cannot pass as two.

## Security posture

| Listener | Code default | Enforced at startup | Authentication | Encryption |
|---|---|---|---|---|
| Orchestrator REST / gRPC | 127.0.0.1:8080 / :9090 | Go `config.CheckBindAddress`: public IP or hostname → refuse; 0.0.0.0 → warning | none | none |
| Worker Health | 127.0.0.1:50100 | `worker/config/bind_policy.py`: same rules | none | none |
| StageExecution | off; 127.0.0.1:50200 | same, plus 0.0.0.0 needs `execution_allow_all_interfaces` | none | none |

- **Phase 8: who connected.** The orchestrator records each
  registration's TCP peer IP (`observed_peer_ip`) and uses it in
  duplicate-id detection. Heartbeats are still matched by id only.
- **Unprotected:**
  - anyone who can reach a LAN-bound port can register workers or run
    stages;
  - traffic can be read and altered on the network path;
  - ids, hostnames and plans are self-asserted;
  - the peer IP is what TCP saw, which a device on the same LAN can
    spoof.
- **Future work, not implemented:** gRPC TLS with mutual authentication
  (per-machine certificates from a private CA), or server-authenticated
  TLS plus a bearer token checked in an interceptor. These would use
  standard gRPC/TLS libraries only, **no custom cryptography**.
