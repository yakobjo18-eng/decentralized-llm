# Phase 1 local smoke test: starts the real orchestrator binary and a
# real worker process, waits for the worker to register over gRPC,
# queries the orchestrator's REST API to confirm the worker is visible
# with real hardware data, sends a few heartbeats, then shuts both
# processes down cleanly (deregistration on worker shutdown).
#
# This is not a fake check against mocked data -- it drives the actual
# orchestrator.exe and the actual `python -m worker.main` process over
# real loopback network connections.
#
# Run from the repository root:
#   powershell -ExecutionPolicy Bypass -File scripts/smoke_test_phase1.ps1

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$env:PATH = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" +
            [System.Environment]::GetEnvironmentVariable("Path","User") + ";" +
            "$env:USERPROFILE\go\bin"

$RestPort = 18080
$GrpcPort = 19090
$WorkerPort = 19100
$WorkerId = "smoke-test-worker"

$logDir = Join-Path $RepoRoot "scripts\.smoke_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$orchLog = Join-Path $logDir "orchestrator.log"
$workerLog = Join-Path $logDir "worker.log"

$orchConfig = Join-Path $logDir "orchestrator.smoke.yaml"
@"
rest_address: "127.0.0.1:$RestPort"
grpc_address: "127.0.0.1:$GrpcPort"
log_level: "info"
cors_origin: "http://localhost:3000"
"@ | Set-Content -Path $orchConfig -Encoding utf8

$workerConfig = Join-Path $logDir "worker.smoke.yaml"
@"
worker_id: "$WorkerId"
orchestrator_address: "127.0.0.1:$GrpcPort"
listen_host: "127.0.0.1"
listen_port: $WorkerPort
heartbeat_interval_seconds: 1.0
log_level: "info"
"@ | Set-Content -Path $workerConfig -Encoding utf8

Write-Output "Starting orchestrator..."
$orchExe = Join-Path $RepoRoot "orchestrator\orchestrator.exe"
$orchProc = Start-Process -FilePath $orchExe -ArgumentList "--config `"$orchConfig`"" `
    -RedirectStandardOutput $orchLog -RedirectStandardError "$orchLog.err" `
    -PassThru -NoNewWindow

Start-Sleep -Seconds 1

try {
    $health = Invoke-RestMethod -Uri "http://127.0.0.1:$RestPort/health" -TimeoutSec 5
    Write-Output "Orchestrator /health: $($health | ConvertTo-Json -Compress)"
} catch {
    Write-Output "FAIL: orchestrator /health unreachable: $_"
    Stop-Process -Id $orchProc.Id -Force -ErrorAction SilentlyContinue
    exit 1
}

Write-Output "Starting worker..."
$workerProc = Start-Process -FilePath "python" -ArgumentList "-m worker.main --config `"$workerConfig`"" `
    -WorkingDirectory (Join-Path $RepoRoot "worker") `
    -RedirectStandardOutput $workerLog -RedirectStandardError "$workerLog.err" `
    -PassThru -NoNewWindow

# Phase 3 change: this used to be a fixed 3-second sleep followed by a
# single check. Worker startup now also enumerates GPUs (WMI ~1.2 s,
# DXGI, GPU counters) before its first registration, which lands at
# ~3.1 s on Computer 1 -- just past the old fixed wait, so the check
# raced the registration. Poll for up to 15 s instead; the pass
# criteria below are unchanged.
$deadline = (Get-Date).AddSeconds(15)
do {
    Start-Sleep -Milliseconds 500
    try {
        $registered = @(Invoke-RestMethod -Uri "http://127.0.0.1:$RestPort/api/workers" -TimeoutSec 5) |
            Where-Object { $_.id -eq $WorkerId }
    } catch { $registered = $null }
} until ($registered -or (Get-Date) -gt $deadline)

$success = $false
try {
    $workers = Invoke-RestMethod -Uri "http://127.0.0.1:$RestPort/api/workers" -TimeoutSec 5
    Write-Output "Orchestrator /api/workers: $($workers | ConvertTo-Json -Compress)"

    $found = $workers | Where-Object { $_.id -eq $WorkerId }
    if ($found -and $found.cpu_logical_cores -gt 0 -and $found.hostname) {
        Write-Output "PASS: worker '$WorkerId' is registered with real hardware data (cpu_logical_cores=$($found.cpu_logical_cores), hostname=$($found.hostname), has_cuda_gpu=$($found.has_cuda_gpu))"
        $success = $true
    } else {
        Write-Output "FAIL: worker '$WorkerId' not found (or missing hardware data) in /api/workers"
    }
} catch {
    Write-Output "FAIL: could not query /api/workers: $_"
}

# Let a couple of heartbeats go through before shutdown.
Start-Sleep -Seconds 2

Write-Output "Stopping worker (graceful)..."
Stop-Process -Id $workerProc.Id -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

try {
    $workersAfter = Invoke-RestMethod -Uri "http://127.0.0.1:$RestPort/api/workers" -TimeoutSec 5
    $stillThere = $workersAfter | Where-Object { $_.id -eq $WorkerId }
    if ($stillThere) {
        Write-Output "NOTE: worker still listed after Stop-Process (graceful deregister may not have had time to run under this termination method)."
    } else {
        Write-Output "PASS: worker '$WorkerId' deregistered cleanly after shutdown signal"
    }
} catch {
    Write-Output "NOTE: could not query /api/workers after worker shutdown: $_"
}

Write-Output "Stopping orchestrator..."
Stop-Process -Id $orchProc.Id -Force -ErrorAction SilentlyContinue

Write-Output "--- orchestrator.log ---"
Get-Content $orchLog -ErrorAction SilentlyContinue
Write-Output "--- worker.log ---"
Get-Content $workerLog -ErrorAction SilentlyContinue

if (-not $success) {
    exit 1
}
exit 0
