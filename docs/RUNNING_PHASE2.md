# Running Phase 2 (multi-worker, still on one machine)

Everything below has been run and verified on Computer 1 only. This is
local multi-process testing, not physical multi-machine testing --
Computer 2 and Computer 3 are not involved yet. See
docs/NETWORKING.md for what changes once they are.

Phase 1 setup (one-time dependency install) still applies; see
docs/RUNNING_PHASE1.md if you haven't done that yet.

## New in Phase 2

- `heartbeat_timeout_seconds` in `orchestrator.yaml` (default 15s):
  how long a worker can go without a heartbeat before it's reported
  offline.
- `GET /api/workers` now includes `"online": true/false` and
  `"last_heartbeat_age_seconds"` per worker.
- `GET /health` now includes `"online_workers"` alongside
  `"registered_workers"`.
- Registering a worker ID that's already registered and online, from
  a *different* address, is now rejected (not silently overwritten).
  Re-registering from the *same* address (a fast restart), or
  registering after the previous registration has gone offline (a
  genuine reconnect), is still allowed.
- The worker now retries the initial connection to the orchestrator
  with backoff instead of crashing if the orchestrator isn't up yet,
  and handles `SIGBREAK` on Windows (for graceful shutdown of one
  specific background process via `CTRL_BREAK_EVENT`).

## Running three workers locally

Rebuild the orchestrator first (Phase 2 changed it):

```powershell
cd "C:\Users\Yakob\OneDrive\Documents\Programs\decentralized llm\orchestrator"
go build -o orchestrator.exe ./cmd/orchestrator
.\orchestrator.exe --config orchestrator.yaml
```

### Option A: one command, interactive (recommended for a quick look)

```powershell
cd "C:\Users\Yakob\OneDrive\Documents\Programs\decentralized llm"
python scripts\multi_worker_test.py
```

This starts the orchestrator and three workers on test ports, prints
each worker's `online` status every 3 seconds, and prints each
worker's PID. To exercise the failure path yourself, open another
terminal and force-kill one of the printed PIDs:

```powershell
Stop-Process -Id <printed-pid> -Force
```

Watch the printed status: that worker's `online` flips to `false`
within `heartbeat_timeout_seconds` (8s in this script), the other two
stay `online: true` throughout. Press Ctrl+C in the original terminal
to shut everything down gracefully.

### Option B: three separate terminals (to watch each worker's own logs)

Create three config files (adjust ports/IDs as you like):

```powershell
cd "C:\Users\Yakob\OneDrive\Documents\Programs\decentralized llm\worker"

@'
worker_id: "worker-1"
orchestrator_address: "127.0.0.1:9090"
listen_host: "127.0.0.1"
listen_port: 50101
heartbeat_interval_seconds: 5.0
log_level: "info"
'@ | Set-Content worker-1.yaml

@'
worker_id: "worker-2"
orchestrator_address: "127.0.0.1:9090"
listen_host: "127.0.0.1"
listen_port: 50102
heartbeat_interval_seconds: 5.0
log_level: "info"
'@ | Set-Content worker-2.yaml

@'
worker_id: "worker-3"
orchestrator_address: "127.0.0.1:9090"
listen_host: "127.0.0.1"
listen_port: 50103
heartbeat_interval_seconds: 5.0
log_level: "info"
'@ | Set-Content worker-3.yaml
```

Then, in three separate terminals (with the orchestrator already
running from the snippet above):

```powershell
# Terminal A
cd "C:\Users\Yakob\OneDrive\Documents\Programs\decentralized llm\worker"
python -m worker.main --config worker-1.yaml

# Terminal B
cd "C:\Users\Yakob\OneDrive\Documents\Programs\decentralized llm\worker"
python -m worker.main --config worker-2.yaml

# Terminal C
cd "C:\Users\Yakob\OneDrive\Documents\Programs\decentralized llm\worker"
python -m worker.main --config worker-3.yaml
```

Check all three registered:

```powershell
Invoke-RestMethod http://localhost:8080/api/workers | Format-Table id, online, status, last_heartbeat_age_seconds
```

Stop one unexpectedly (simulating a crash, not Ctrl+C) by closing its
terminal window or killing its process directly, then re-check
`/api/workers` after `heartbeat_timeout_seconds` -- it should show
`online: false` while the other two remain `online: true`. Restart it
with the same `python -m worker.main --config worker-N.yaml` command
and it registers again.

## Automated tests

```powershell
# Go: registry conflict/timeout logic, gRPC adapter, REST API
cd "C:\Users\Yakob\OneDrive\Documents\Programs\decentralized llm\orchestrator"
go test ./... -v

# Python: config/hardware/retry unit tests, plus two REAL integration
# tests that build on orchestrator.exe (build it first -- see above).
cd "C:\Users\Yakob\OneDrive\Documents\Programs\decentralized llm\worker"
python -m pytest -v
```

The two integration tests
(`tests/test_graceful_shutdown_integration.py`,
`tests/test_three_worker_integration.py`) actually start
`orchestrator.exe` and real `python -m worker.main` subprocesses; they
are skipped automatically (not failed) if `orchestrator.exe` hasn't
been built yet.

## What this does NOT test

- **Physical multi-machine behavior.** Everything above runs on
  Computer 1 only, over loopback. Computer 2 and Computer 3 have not
  been used. Network-partition behavior across real machines, real
  LAN latency, and real firewall configuration are untested -- see
  docs/NETWORKING.md for what's believed to be ready and what isn't
  verified.
- **Redis / persistence.** Still deferred, as decided after Phase 1.
- **Scheduling.** There is still no scheduler (Phase 4/7); `online`
  is the signal a future scheduler is expected to filter on, but
  nothing consumes it yet besides the REST API and these tests.
