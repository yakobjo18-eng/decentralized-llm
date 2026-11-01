# Phase 8 LAN test -- 2026-11-01T13:40:46

- **Result:** FAIL (failed: P5, P7)
- **Classification:** `REAL_MULTI_PROCESS`
- Coordinator: `YakobsPC`, Windows-11-10.0.26200-SP0, Python 3.13.3
- Commit: `b1ab6acaaf97fd0b98026d13d45aef3c8c78d1bb` on `main` -- **uncommitted changes present**
- Orchestrator: http://127.0.0.1:56936; endpoints: `rehearsal-a=127.0.0.1:56938`, `rehearsal-b=127.0.0.1:56940`
- Plan source: REHEARSAL: live registrations were INFEASIBLE on this machine; plan built from the same REAL records with SYNTHETIC capacity -- rehearsal only
- Timings are single-run wall clock, not benchmarks.

| Step | Title | Result | Time | Detail |
|---|---|---|---|---|
| P1 | coordinator environment (host, IPs, commit) | **PASS** | 0.09s | coordinator YakobsPC, commit b1ab6acaaf97fd0b98026d13d45aef3c8c78d1bb on main (UNCOMMITTED changes present: record them) |
| P2 | wait for registrations and backend probes | **PASS** | 4.21s | all 2 workers registered, online, backend probes finished (waited 4.2s) |
| P3 | preflight: orchestrator, registrations, endpoints, protocol | **PASS** | 0.30s | {'PASS': 25, 'WARN': 0, 'FAIL': 0, 'SKIP': 1, 'INFO': 8} |
| P4 | plan the tiny model across the endpoint workers | **PASS** | 7.54s | REHEARSAL: live registrations were INFEASIBLE on this machine; plan built from the same REAL records with SYNTHETIC capacity -- rehearsal only; plan 65a9f640f092c652...; stages [{'stage': 0, 'worker': 'rehearsal-a', 'host': 'YakobsPC', 'units': [0, 3]}, {'stage': 1, 'worker': 'rehearsal-b', 'host': 'YakobsPC', 'units': [3, 7]}] |
| P5 | preflight: plan, assignment, limits, multi-host | **FAIL** | 0.16s | {'PASS': 16, 'WARN': 1, 'FAIL': 1, 'SKIP': 0, 'INFO': 8}; FAILS: ["plan C:\\Users\\Yakob\\OneDrive\\Documents\\Programs\\decentralized llm\\docs\\validation\\phase8-loopback-strict-2026-11-01\\plan.json: hosts spanned: 1 host(s): ['YakobsPC']"] |
| P6 | execute the plan remotely x1, compare with the local reference | **PASS** | 0.23s | run 1: succeeded, diff 0.0, 0.2256s (single runs, not a benchmark) |
| P7 | multi-machine criterion (evidence-based classification) | **FAIL** | 0.00s | classification REAL_MULTI_PROCESS: 1 reported hostname(s) ['YakobsPC'], 0 non-loopback registration peer IP(s) [] -- multi-machine criterion NOT met |
| P8 | disconnect/restart | **SKIP** | 0.00s | not run: no --interactive operator |

## Evidence per worker

| Worker | Reported host | Reported pid | Endpoint | Orchestrator saw it register from |
|---|---|---|---|---|
| rehearsal-a | YakobsPC | 8012 | 127.0.0.1:56938 | 127.0.0.1 |
| rehearsal-b | YakobsPC | 23516 | 127.0.0.1:56940 | 127.0.0.1 |

Files: `preflight-live.txt`, `preflight-plan.txt`, `plan.json`, `result-*.json`, `report.json`.
