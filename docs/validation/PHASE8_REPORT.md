# Phase 8 validation report: two-machine LAN readiness (2026-11-01)

**Bottom line.** The repository is ready for the first real two-computer
test, and everything that can be tested on one machine passed. **No
separate physical computer took part in Phase 8. No result here is
`REAL_MULTI_MACHINE`, and the physical test remains `PENDING`.**

All runs: Computer 1 (`YakobsPC`), Windows 11, Python 3.13.3, branch
`main` at `b1ab6ac` plus the uncommitted Phase 8 changes.

## What changed and why

| Change | Why (what a real two-machine test would have hit) |
|---|---|
| The orchestrator records each registration's **TCP peer IP** (`observed_peer_ip`), uses it in duplicate-id detection, and exposes it in REST | With Health on loopback (the least-privilege setting), every machine reports `127.0.0.1:50100`. A duplicated `worker_id` on Computer 2 would have been accepted as a "refresh" and silently taken over Computer 1's worker. The peer IP is also the strongest evidence of which machine a worker runs on. |
| Preflight `--orchestrator-config` | Computer 1's orchestrator config had no check: bind policy, address ownership, port, simulated workers. |
| Preflight **local firewall check** (`netsh`, read-only) | The usual first-test failure is a missing rule, or a Block rule left by a dismissed prompt. The PowerShell cmdlets took **65 s** on Computer 1; `netsh` took **0.4 s**. |
| Preflight **port owner** (psutil) | On Computer 1, **port 8080 (the orchestrator's REST default) was held by an unrelated Java process**. "In use" alone could not tell that apart from "already running". |
| Preflight `--probe-orchestrator` | Lets Computer 2 test that it reaches Computer 1's gRPC port *before* starting its worker. |
| `scripts/phase8_lan.py` (`orchestrator-config`, `worker-config`, `run`, `rehearse`) | Configs from command-line values only, validated. The test itself: plans from live registrations only, repeats execution, and judges the classification from evidence. |
| Redaction of local IPs in rehearsal reports | Rehearsal artifacts may be committed. A first recording leaked this machine's LAN address through the second preflight file; it was fixed, a test now scans every artifact, and the runs were re-recorded. |

## Commands run and actual results

| # | Command | Result | Classification |
|---|---|---|---|
| 1 | `cd orchestrator; gofmt -l .; go vet ./...; go test ./... -v -count=1` | gofmt and vet clean; **111 `--- PASS`, 0 `--- FAIL`** (104 before + 7 new) | in-process Go tests (`REAL_SINGLE_MACHINE`); not individually classified |
| 2 | `cd worker; python -m pytest -q` | **150 passed**, 0 failed | 57 REAL_SINGLE_MACHINE, 3 REAL_MULTI_PROCESS, 90 SIMULATED |
| 3 | `cd model-partitioner; python -m pytest -q` | **383 passed**, 0 failed (326 before + 57 new) | 233 REAL_SINGLE_MACHINE, 14 REAL_MULTI_PROCESS, 0 REAL_MULTI_MACHINE, 136 SIMULATED |
| 4 | `powershell -ExecutionPolicy Bypass -File scripts/smoke_test_phase1.ps1` | **PASS** | REAL_MULTI_PROCESS |
| 5 | `python scripts/phase7_local_validation.py --out <scratch>` (regression after the preflight changes) | **14/14** | REAL_MULTI_PROCESS |
| 6 | `python scripts/phase8_lan.py rehearse --out docs/validation/phase8-loopback-rehearsal-2026-11-01 --repeat 3` | **RESULT PASS**, P1–P8 PASS | **REAL_MULTI_PROCESS** |
| 7 | `python scripts/phase8_lan.py rehearse --strict --no-disruption --repeat 1 --out docs/validation/phase8-loopback-strict-2026-11-01` | **RESULT FAIL, as designed**: P5 `hosts spanned: 1`, P7 `multi-machine criterion NOT met` | REAL_MULTI_PROCESS |

### New tests

- **Go (7):**
  - registry, 4 tests: a same-address registration from another peer
    conflicts; same peer is a refresh; an unknown peer keeps the old
    behaviour; an offline id may return from a new peer;
  - gRPC server, 2 tests: the peer IP is recorded over a **real gRPC
    connection**, and left empty without peer info;
  - REST, 1 test: `observed_peer_ip` appears next to the reported
    address.
- **`model-partitioner/tests/test_phase8_lan.py` (55):**
  - firewall evaluation on canned `netsh` output (24, `SIMULATED`):
    no rule, restricted allow, broad allow, Block wins, non-matching
    rules ignored, firewall off, Public-profile program rules,
    unparseable output means SKIP;
  - a live `netsh` read on this machine;
  - orchestrator config checks, including the env override as in Go;
  - **Go/Python parity**: the real `orchestrator.exe` refuses exactly
    what preflight FAILs;
  - the `--probe-orchestrator` startup-order hint;
  - config generators: round-trip through the worker loader, refusal of
    public, `0.0.0.0`, hostname, loopback-without-flag, bad id and port
    clash, no overwrite, and the generated orchestrator config starting
    the real binary;
  - `judge()`: two machines pass; renamed hosts on one machine fail; one
    multi-homed machine fails; the rehearsal flag never upgrades the
    class;
  - three full loopback rehearsals: end-to-end, strict-fails, and no LAN
    address in any artifact (REAL_MULTI_PROCESS).
- **`test_preflight.py` (+2):** a port held by an unrelated program
  FAILs; a port held by `orchestrator.exe` is INFO. The existing
  in-use test now asserts the owner, which is stricter.

### Failures during development (all fixed; none open)

1. **Rehearsal race (script bug).** A worker logs "StageExecution
   listening" before it registers, and the first `run` ordering ran
   preflight in that gap. That passed from the CLI by luck, and failed
   under pytest. Fix: P2 now waits for registration and backend probes,
   and P3 preflight follows (it still runs on timeout, to explain why).
   It then passed twice in a row and in every full run since.
2. **IP leak in rehearsal artifacts.** The redaction covered only the
   first preflight file. Fixed in every preflight write, with a test
   that scans all files.
3. **One existing assertion changed its text.** It now names the owner
   of the port instead of saying "in use"; the status is the same
   (WARN), and the assertion is stricter.

## Findings on Computer 1 (not modified; for the operator)

- **Firewall exposure.** Inbound Allow rules on the **Public** profile
  admit `python.exe` (2 rules, "Python") and `orchestrator.exe` (2
  rules) on any port. They are inactive on the current Private network.
- **No Private-profile rule exists yet** for 9090 or 50201. This is
  expected; the procedure creates them.
- **Port 8080** is held by `java.exe` (Eclipse Adoptium JDK) on
  `0.0.0.0:8080`. Use another `--rest-port` while it runs.
- **Windows excluded TCP range 50000–50059.** The project's ports are
  outside it.
- **Live planning is infeasible on Computer 1 under the default
  margins** (52–82 MiB effective vs. 700 MiB per stage in Phase 7). The
  real test may need freed RAM or a recorded margin deviation.

## Remaining manual steps for the first real LAN test

1. Bring Computer 2 onto the same private subnet. Check out the same
   commit; install Python 3.13 and the requirements.
2. Write down `C1_IP` and `C2_IP` (DEPLOYMENT_LAN.md, Step 2).
3. Generate the configs with `scripts/phase8_lan.py` (Step 3).
4. Add the two least-privilege firewall rules (Step 4).
5. Run preflight on both machines (Step 5), then start the services in
   order (Step 6), including `--probe-orchestrator` on Computer 2.
6. Run `python scripts/phase8_lan.py run … --interactive --out
   docs/validation/phase8-lan-<date>` on Computer 1 (Step 7). It must end
   with `RESULT PASS -- classification REAL_MULTI_MACHINE`.
7. Fill in the results table in DEPLOYMENT_LAN.md, then shut down and
   remove the rules (Step 8).
