# Regenerates Go and Python code from shared/proto/*.proto.
#
# Prerequisites (Phase 1 installs these once):
#   - Go toolchain, with protoc-gen-go and protoc-gen-go-grpc installed:
#       go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
#       go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
#     (installed to $env:USERPROFILE\go\bin -- must be on PATH)
#   - Python worker virtualenv with grpcio-tools installed (provides the
#     protoc binary itself -- no separate protoc.exe install needed):
#       pip install -r worker/requirements.txt
#
# Run from the repository root:
#   powershell -ExecutionPolicy Bypass -File scripts/generate_protos.ps1

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$env:PATH = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" +
            [System.Environment]::GetEnvironmentVariable("Path","User") + ";" +
            "$env:USERPROFILE\go\bin"

$protoFiles = @(
    "shared/proto/common.proto",
    "shared/proto/registry.proto",
    "shared/proto/health.proto"
)

Write-Output "Generating Go code -> orchestrator/internal/proto/..."
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue "orchestrator/internal/proto/common","orchestrator/internal/proto/registry","orchestrator/internal/proto/health"
python -m grpc_tools.protoc -I shared/proto `
    --go_out=. --go_opt=paths=import `
    --go-grpc_out=. --go-grpc_opt=paths=import `
    @protoFiles
if ($LASTEXITCODE -ne 0) { throw "Go proto generation failed" }

Write-Output "Generating Python code -> worker/worker/generated/..."
python -m grpc_tools.protoc -I shared/proto `
    --python_out=worker/worker/generated `
    --grpc_python_out=worker/worker/generated `
    --pyi_out=worker/worker/generated `
    @protoFiles
if ($LASTEXITCODE -ne 0) { throw "Python proto generation failed" }

Write-Output "Done."
