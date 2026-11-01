# Phase 7 local validation -- 2026-11-01T12:53:41

Host `YakobsPC`, Windows-11-10.0.26200-SP0, Python 3.13.3. All steps ran on this ONE computer over 127.0.0.1. **No step is REAL_MULTI_MACHINE.** Timings are single-run wall clock, not benchmarks.

**13/14 steps passed.**

| Step | Title | Class | Result | Time | Detail |
|---|---|---|---|---|---|
| S01 | preflight: runtime and the three worker configs | REAL_SINGLE_MACHINE | **PASS** | 0.21s | {'PASS': 39, 'WARN': 0, 'FAIL': 0, 'SKIP': 3, 'INFO': 11} (worker configs + runtime, before start) |
| S02 | start orchestrator + 3 worker.main processes (execution enabled, loopback) | REAL_MULTI_PROCESS | **PASS** | 10.61s | orchestrator healthy in 0.10s; workers ready (StageExecution listening) in val-a 9.7s, val-c 9.7s, val-b 9.7s; all online+REAL |
| S03 | listener exposure audit (psutil, actual bound addresses) | REAL_MULTI_PROCESS | **PASS** | 0.00s | 8 listening sockets, all on 127.0.0.1 |
| S04 | plan from live orchestrator registrations | REAL_MULTI_PROCESS | **PASS** | 11.20s | waited 4.1s for backend probes; INFEASIBLE from live registrations: The whole model on one worker needs 700.40 MiB; the largest effective capacity is 51.81 MiB (val-b); all usable workers together have 71.63 MiB.; Unit block.3 alone needs 700.20 MiB (including the per-worker runtime overhead), more than any worker can hold.; Every stage also pays a 700.00 MiB runtim -- pipelines below use these REAL records with a SYNTHETIC 3 GiB usable capacity, for planning only |
| S05 | preflight: live orchestrator + endpoints + 2-stage plan | REAL_MULTI_PROCESS | **PASS** | 0.14s | {'PASS': 16, 'WARN': 1, 'FAIL': 0, 'SKIP': 0, 'INFO': 9} |
| S06 | 2-worker remote pipeline | REAL_MULTI_PROCESS | **PASS** | 0.23s | succeeded, output [2, 8, 128], next tokens [60, 96], reference diff 0.0, 2 processes on ['YakobsPC'], total 0.234s |
| S07 | 3-worker remote pipeline | REAL_MULTI_PROCESS | **PASS** | 0.25s | succeeded, output [1, 8, 128], next tokens [60], reference diff 0.0, 3 processes on ['YakobsPC'], total 0.251s |
| S08 | resource-limit failure (val-c max_batch 1, request batch 2) | REAL_MULTI_PROCESS | **PASS** | 0.21s | failed at stage 0: LIMIT_EXCEEDED: ['batch 2 > worker limit 1'] |
| S09 | malformed requests against a live worker process | REAL_MULTI_PROCESS | **FAIL** | 0.25s | protocol_version 0 -> PROTOCOL_VERSION, missing plan_json -> MALFORMED_REQUEST, corrupt tensor bytes -> UNASSIGNED_STAGE, unassigned stage (stage 1 sent to val-a) -> INVALID_TENSOR, no deadline -> DEADLINE; worker still serves afterwards: True |
| S10 | request deadline (20 ms stage timeout) | REAL_MULTI_PROCESS | **PASS** | 0.22s | failed: DEADLINE (only 0.020s left before the deadline (< 0.05s minimum)); attempts ['STAGE_STATUS_REJECTED'] -- not retried |
| S11 | graceful worker stop (CTRL_BREAK) and restart | REAL_MULTI_PROCESS | **PASS** | 11.31s | exit code 0, 'worker shutdown complete' logged: True, deregistered in 0.0s; while down: failed RETRIES_EXHAUSTED at stage 0 (executed False); restarted+online in 8.0s; after: succeeded, pid 40144 -> 21284 |
| S12 | hard kill of a worker, offline detection, restart | REAL_MULTI_PROCESS | **PASS** | 11.13s | killed; orchestrator marked it offline after 2.6s (heartbeat timeout 3s); restarted, online after 0.8s; pipeline succeeded |
| S13 | orchestrator restart: execution while down, worker re-registration | REAL_MULTI_PROCESS | **PASS** | 1.51s | pipeline while the orchestrator was DOWN: succeeded (execution does not depend on it); after restart all 3 workers re-registered within 1.1s (logged: {'val-a': True, 'val-b': True, 'val-c': True}); pipeline succeeded |
| S15 | clean shutdown of all workers (deregistration) | REAL_MULTI_PROCESS | **PASS** | 2.90s | {'val-a': {'exit_code': 0, 'clean': True}, 'val-b': {'exit_code': 0, 'clean': True}, 'val-c': {'exit_code': 0, 'clean': True}}; orchestrator lists no workers after 0.0s |

Logs: `logs/` (one file per process). Evidence for every step: `report.json`.
