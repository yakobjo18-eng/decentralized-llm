# Running Phase 1 (local, single machine)

Everything below has been run and verified on Computer 1 (Windows 11,
Python 3.13.3, Go 1.27.0). Phase 1 does not yet involve Computer 2 or
Computer 3 -- that starts in Phase 2.

For running multiple workers (Phase 2: heartbeat timeouts, duplicate-ID
handling, a real three-worker test), see docs/RUNNING_PHASE2.md
instead. This document still applies as-is for the single
orchestrator + single worker setup described below.

## One-time setup

```powershell
# From the repository root.
cd "C:\Users\Yakob\OneDrive\Documents\Programs\decentralized llm"

# Go dependencies (already fetched during Phase 1; re-run is harmless).
cd orchestrator
go mod tidy
cd ..

# Python dependencies.
cd worker
python -m pip install -r requirements.txt
cd ..

# Frontend dependencies.
cd frontend
npm install
cd ..
```

If `go`, `python`, or `npm` are not recognized, open a new PowerShell
window (installers update PATH at the registry level; an already-open
shell may not see it) or run:

```powershell
$env:PATH = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
```

## Configuration

Copy the example configs and adjust if needed (defaults work for local,
single-machine use as-is):

```powershell
Copy-Item orchestrator\orchestrator.example.yaml orchestrator\orchestrator.yaml
Copy-Item worker\worker.example.yaml worker\worker.yaml
Copy-Item .env.example .env
Copy-Item frontend\.env.example frontend\.env.local
```

## Running the three processes

Open three separate PowerShell windows.

**Terminal 1 -- Orchestrator:**

```powershell
cd "C:\Users\Yakob\OneDrive\Documents\Programs\decentralized llm\orchestrator"
go build -o orchestrator.exe ./cmd/orchestrator
.\orchestrator.exe --config orchestrator.yaml
```

You should see:

```
gRPC WorkerRegistry service listening on 127.0.0.1:9090
REST API listening on 127.0.0.1:8080
```

(Before Phase 7 the defaults were `0.0.0.0`; they are loopback now. Set
`grpc_address` / `rest_address` to a private LAN IP to serve other machines.)

**Terminal 2 -- Worker:**

```powershell
cd "C:\Users\Yakob\OneDrive\Documents\Programs\decentralized llm\worker"
python -m worker.main --config worker.yaml
```

You should see the worker collect its hardware snapshot, register, and
start sending heartbeats every 5 seconds. Press Ctrl+C to stop it
(this triggers deregistration).

**Terminal 3 -- Frontend (optional for Phase 1; full dashboard is Phase 9):**

```powershell
cd "C:\Users\Yakob\OneDrive\Documents\Programs\decentralized llm\frontend"
npm run dev
```

Open http://localhost:3000 -- you should see the worker you started in
Terminal 2 listed with its real hardware (CPU core count, RAM, and
"None (CPU-only)" for GPU on a machine with no CUDA device).

## Verifying without the frontend

```powershell
Invoke-RestMethod http://localhost:8080/health
Invoke-RestMethod http://localhost:8080/api/workers
```

## Automated checks

```powershell
# Go unit tests
cd "C:\Users\Yakob\OneDrive\Documents\Programs\decentralized llm\orchestrator"
go test ./...

# Python unit/integration tests
cd "C:\Users\Yakob\OneDrive\Documents\Programs\decentralized llm\worker"
python -m pytest -v

# Full local smoke test: builds nothing extra, starts the real
# orchestrator.exe and a real worker process on test ports, verifies
# registration and heartbeats via the REST API, then shuts both down.
cd "C:\Users\Yakob\OneDrive\Documents\Programs\decentralized llm"
powershell -ExecutionPolicy Bypass -File scripts\smoke_test_phase1.ps1
```

All three passed on Computer 1 as of Phase 1 (19 Go tests, 9 Python
tests, smoke test PASS). Test counts have grown since Phase 2 -- see
docs/DEVELOPMENT_STATUS.md for the current, exact numbers and run log.

## Regenerating protobuf code

Only needed after editing a file in `shared/proto/`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\generate_protos.ps1
```

## What this does NOT yet do

- No model loading, no partitioning, no actual inference. The worker
  only registers and heartbeats; it does not load any Transformer
  layers yet (Phase 4/6).
- No multi-worker testing, no heartbeat-timeout detection (Phase 2/8).
- The frontend page is a one-page connectivity check, not the full
  7-page dashboard (Phase 9).
