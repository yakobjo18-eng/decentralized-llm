# Recorded validation runs

Each directory is one run of a validation script, kept as evidence. Each
contains `report.md` (summary), `report.json` (every step's evidence),
`logs/` (one log per process) and `work/` (the generated configs and
plans).

| Run | Script | Machine(s) | Classification | Result |
|---|---|---|---|---|
| [phase7-local-2026-11-01-run1](phase7-local-2026-11-01-run1/report.md) | `scripts/phase7_local_validation.py` | Computer 1 only | REAL_MULTI_PROCESS | 13/14. S09 failed because of a **validation-script bug**: it assumed `val-a` owned stage 0, but the planner put `val-b` first. The worker's answers were correct. Fixed in the script. |
| [phase7-local-2026-11-01-run2](phase7-local-2026-11-01-run2/report.md) | same, after the fix | Computer 1 only | REAL_MULTI_PROCESS | **14/14** |
| [phase8-loopback-rehearsal-2026-11-01](phase8-loopback-rehearsal-2026-11-01/report.md) | `scripts/phase8_lan.py rehearse` | Computer 1 only | REAL_MULTI_PROCESS | **PASS** (P1–P8). Never claims multi-machine. |
| [phase8-loopback-strict-2026-11-01](phase8-loopback-strict-2026-11-01/report.md) | `scripts/phase8_lan.py rehearse --strict` | Computer 1 only | REAL_MULTI_PROCESS | **FAIL by design**: one machine cannot pass the multi-machine criterion |
| Physical two-computer test | `scripts/phase8_lan.py run` (DEPLOYMENT_LAN.md, Step 7) | Computer 1 + Computer 2 | REAL_MULTI_MACHINE | **PENDING**: no second computer has taken part yet |

Phase 8 summary, commands and test counts: [PHASE8_REPORT.md](PHASE8_REPORT.md).
Rehearsal artifacts redact this machine's non-loopback IP addresses. A real
`run` keeps them, because they are the evidence.

Timings in these reports are single-run wall clock on a loaded laptop,
not benchmarks.
