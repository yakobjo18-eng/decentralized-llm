# Recorded validation runs

Each directory is one run of a validation script, kept as evidence. Each
contains `report.md` (summary), `report.json` (every step's evidence),
`logs/` (one log per process) and `work/` (the generated configs and
plans).

| Run | Script | Machine(s) | Classification | Result |
|---|---|---|---|---|
| [phase7-local-2026-11-01-run1](phase7-local-2026-11-01-run1/report.md) | `scripts/phase7_local_validation.py` | Computer 1 only | REAL_MULTI_PROCESS | 13/14. S09 failed because of a **validation-script bug**: it assumed `val-a` owned stage 0, but the planner put `val-b` first. The worker's answers were correct. Fixed in the script. |
| [phase7-local-2026-11-01-run2](phase7-local-2026-11-01-run2/report.md) | same, after the fix | Computer 1 only | REAL_MULTI_PROCESS | **14/14** |
| Physical two-computer test | docs/DEPLOYMENT_LAN.md checklist | Computer 1 + Computer 2 | REAL_MULTI_MACHINE | **PENDING**: Computer 2 is unavailable |

Timings in these reports are single-run wall clock on a loaded laptop,
not benchmarks.
