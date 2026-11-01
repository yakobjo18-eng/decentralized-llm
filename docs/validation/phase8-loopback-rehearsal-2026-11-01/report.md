# Phase 8 LAN test -- 2026-11-01T13:40:03

- **Result:** PASS
- **Classification:** `REAL_MULTI_PROCESS` (rehearsal: --allow-single-host)
- Coordinator: `YakobsPC`, Windows-11-10.0.26200-SP0, Python 3.13.3
- Commit: `b1ab6acaaf97fd0b98026d13d45aef3c8c78d1bb` on `main` -- **uncommitted changes present**
- Orchestrator: http://127.0.0.1:56775; endpoints: `rehearsal-a=127.0.0.1:56785`, `rehearsal-b=127.0.0.1:56787`
- Plan source: REHEARSAL: live registrations were INFEASIBLE on this machine; plan built from the same REAL records with SYNTHETIC capacity -- rehearsal only
- Timings are single-run wall clock, not benchmarks.

| Step | Title | Result | Time | Detail |
|---|---|---|---|---|
| P1 | coordinator environment (host, IPs, commit) | **PASS** | 0.05s | coordinator YakobsPC, commit b1ab6acaaf97fd0b98026d13d45aef3c8c78d1bb on main (UNCOMMITTED changes present: record them) |
| P2 | wait for registrations and backend probes | **PASS** | 4.18s | all 2 workers registered, online, backend probes finished (waited 4.2s) |
| P3 | preflight: orchestrator, registrations, endpoints, protocol | **PASS** | 0.25s | {'PASS': 25, 'WARN': 0, 'FAIL': 0, 'SKIP': 1, 'INFO': 8} |
| P4 | plan the tiny model across the endpoint workers | **PASS** | 7.53s | REHEARSAL: live registrations were INFEASIBLE on this machine; plan built from the same REAL records with SYNTHETIC capacity -- rehearsal only; plan d3f05553cd290dc5...; stages [{'stage': 0, 'worker': 'rehearsal-b', 'host': 'YakobsPC', 'units': [0, 3]}, {'stage': 1, 'worker': 'rehearsal-a', 'host': 'YakobsPC', 'units': [3, 7]}] |
| P5 | preflight: plan, assignment, limits | **PASS** | 0.10s | {'PASS': 16, 'WARN': 1, 'FAIL': 0, 'SKIP': 0, 'INFO': 9} |
| P6 | execute the plan remotely x3, compare with the local reference | **PASS** | 0.70s | run 1: succeeded, diff 0.0, 0.2275s; run 2: succeeded, diff 0.0, 0.2299s; run 3: succeeded, diff 0.0, 0.2327s (single runs, not a benchmark) |
| P7 | multi-machine criterion (evidence-based classification) | **PASS** | 0.00s | classification REAL_MULTI_PROCESS: 1 reported hostname(s) ['YakobsPC'], 0 non-loopback registration peer IP(s) [] -- REHEARSAL (--allow-single-host): multi-machine criterion NOT met and NOT claimed |
| P8 | disconnect/restart: worker stop, restart, orchestrator restart | **PASS** | 18.07s | stopped rehearsal-a: offline after 0.0s; execution failed (RETRIES_EXHAUSTED, executed=False); restarted: ready after 3.6s; execution succeeded; orchestrator restarted: all workers re-registered within 1.0s; execution succeeded |

## Evidence per worker

| Worker | Reported host | Reported pid | Endpoint | Orchestrator saw it register from |
|---|---|---|---|---|
| rehearsal-b | YakobsPC | 15292 | 127.0.0.1:56787 | 127.0.0.1 |
| rehearsal-a | YakobsPC | 21336 | 127.0.0.1:56785 | 127.0.0.1 |

Files: `preflight-live.txt`, `preflight-plan.txt`, `plan.json`, `result-*.json`, `report.json`.
