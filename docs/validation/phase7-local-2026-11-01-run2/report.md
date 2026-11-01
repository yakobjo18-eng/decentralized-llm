# Phase 7 local validation -- 2026-11-01T12:54:56

Host `YakobsPC`, Windows-11-10.0.26200-SP0, Python 3.13.3. All steps ran on this ONE computer over 127.0.0.1. **No step is REAL_MULTI_MACHINE.** Timings are single-run wall clock, not benchmarks.

**14/14 steps passed.**

| Step | Title | Class | Result | Time | Detail |
|---|---|---|---|---|---|
| S01 | preflight: runtime and the three worker configs | REAL_SINGLE_MACHINE | **PASS** | 0.37s | {'PASS': 39, 'WARN': 0, 'FAIL': 0, 'SKIP': 3, 'INFO': 11} (worker configs + runtime, before start) |
| S02 | start orchestrator + 3 worker.main processes (execution enabled, loopback) | REAL_MULTI_PROCESS | **PASS** | 11.09s | orchestrator healthy in 0.10s; workers ready (StageExecution listening) in val-a 10.1s, val-b 10.2s, val-c 10.2s; all online+REAL |
| S03 | listener exposure audit (psutil, actual bound addresses) | REAL_MULTI_PROCESS | **PASS** | 0.00s | 8 listening sockets, all on 127.0.0.1 |
| S04 | plan from live orchestrator registrations | REAL_MULTI_PROCESS | **PASS** | 11.00s | waited 4.0s for backend probes; INFEASIBLE from live registrations: The whole model on one worker needs 700.40 MiB; the largest effective capacity is 54.35 MiB (val-b); all usable workers together have 92.96 MiB.; Unit block.3 alone needs 700.20 MiB (including the per-worker runtime overhead), more than any worker can hold.; Every stage also pays a 700.00 MiB runtim -- pipelines below use these REAL records with a SYNTHETIC 3 GiB usable capacity, for planning only |
| S05 | preflight: live orchestrator + endpoints + 2-stage plan | REAL_MULTI_PROCESS | **PASS** | 0.09s | {'PASS': 16, 'WARN': 1, 'FAIL': 0, 'SKIP': 0, 'INFO': 9} |
| S06 | 2-worker remote pipeline | REAL_MULTI_PROCESS | **PASS** | 0.24s | succeeded, output [2, 8, 128], next tokens [60, 96], reference diff 0.0, 2 processes on ['YakobsPC'], total 0.239s |
| S07 | 3-worker remote pipeline | REAL_MULTI_PROCESS | **PASS** | 0.25s | succeeded, output [1, 8, 128], next tokens [60], reference diff 0.0, 3 processes on ['YakobsPC'], total 0.248s |
| S08 | resource-limit failure (val-c max_batch 1, request batch 2) | REAL_MULTI_PROCESS | **PASS** | 0.25s | failed at stage 2: LIMIT_EXCEEDED: ['batch 2 > worker limit 1'] |
| S09 | malformed requests against a live worker process | REAL_MULTI_PROCESS | **PASS** | 0.26s | sent to val-b: protocol_version 0 -> PROTOCOL_VERSION, missing plan_json -> MALFORMED_REQUEST, corrupt tensor bytes -> INVALID_TENSOR, unassigned stage (stage 1 sent to val-b) -> UNASSIGNED_STAGE, no deadline -> DEADLINE; worker still serves afterwards: True |
| S10 | request deadline (20 ms stage timeout) | REAL_MULTI_PROCESS | **PASS** | 0.21s | failed: DEADLINE (only 0.020s left before the deadline (< 0.05s minimum)); attempts ['STAGE_STATUS_REJECTED'] -- not retried |
| S11 | graceful worker stop (CTRL_BREAK) and restart | REAL_MULTI_PROCESS | **PASS** | 11.58s | exit code 0, 'worker shutdown complete' logged: True, deregistered in 0.0s; while down: failed RETRIES_EXHAUSTED at stage 0 (executed False); restarted+online in 8.1s; after: succeeded, pid 4544 -> 11464 |
| S12 | hard kill of a worker, offline detection, restart | REAL_MULTI_PROCESS | **PASS** | 11.27s | killed; orchestrator marked it offline after 2.8s (heartbeat timeout 3s); restarted, online after 0.5s; pipeline succeeded |
| S13 | orchestrator restart: execution while down, worker re-registration | REAL_MULTI_PROCESS | **PASS** | 1.29s | pipeline while the orchestrator was DOWN: succeeded (execution does not depend on it); after restart all 3 workers re-registered within 0.8s (logged: {'val-a': True, 'val-b': True, 'val-c': True}); pipeline succeeded |
| S15 | clean shutdown of all workers (deregistration) | REAL_MULTI_PROCESS | **PASS** | 2.95s | {'val-a': {'exit_code': 0, 'clean': True}, 'val-b': {'exit_code': 0, 'clean': True}, 'val-c': {'exit_code': 0, 'clean': True}}; orchestrator lists no workers after 0.0s |

Logs: `logs/` (one file per process). Evidence for every step: `report.json`.
