# Decentralized Edge-Compute Pipeline for Large Model Inference

A distributed AI inference framework: multiple consumer-grade
computers, each loading only a subset of a Large Language Model's
Transformer layers, collaborating via pipeline parallelism to run a
model too large for any one of them.

**Status: Phase 1 of 10 complete.** See docs/DEVELOPMENT_STATUS.md for
the live tracker and docs/ARCHITECTURE.md for the full design,
including honest limitations (distributed inference is not faster than
single-machine when a model already fits on one box; mid-generation
worker failure means restarting that request, not seamless recovery).

## What exists right now

- A Go orchestrator (`orchestrator/`) with a WorkerRegistry gRPC
  service (Register/Heartbeat/Deregister) and a REST API
  (`GET /health`, `GET /api/workers`), backed by a real in-memory
  registry behind a storage interface (swappable for Redis later
  without touching callers).
- A Python worker (`worker/`) that collects a real (not fabricated)
  minimal hardware snapshot, registers with the orchestrator over
  gRPC, and sends heartbeats.
- Shared Protocol Buffer definitions (`shared/proto/`) compiled to both
  Go and Python.
- A Next.js frontend skeleton (`frontend/`) with a one-page
  connectivity check against the orchestrator's live REST API -- the
  full 7-page dashboard is Phase 9.
- 19 Go tests, 9 Python tests, and a local smoke test
  (`scripts/smoke_test_phase1.ps1`) that runs the real orchestrator
  binary and a real worker process together and checks the result.

## What does not exist yet

Model loading, model partitioning, tensor transport, and actual
distributed inference. The worker currently does nothing but announce
itself and its hardware -- see docs/ARCHITECTURE.md's phase roadmap for
when each piece lands (Phases 4-6).

## Quick start

See docs/RUNNING_PHASE1.md for exact commands. Summary: run the
orchestrator, then a worker, then open the frontend -- all on one
machine for now.

## Repository layout

```
orchestrator/        Go control plane (registry, REST API, gRPC server)
worker/               Python worker node (hardware profiling, gRPC client)
model-partitioner/    Not yet implemented (Phase 4)
shared/proto/         Protocol Buffer definitions, source of truth for both languages
frontend/             Next.js dashboard
benchmarks/           Not yet implemented (Phase 8)
scripts/              Codegen and smoke-test scripts
docs/                 Architecture, status, running, and networking docs
```

## Documentation

- docs/ARCHITECTURE.md -- Phase 0 design, risk analysis, and roadmap.
- docs/DEVELOPMENT_STATUS.md -- live phase/test status tracker.
- docs/RUNNING_PHASE1.md -- how to build and run everything that
  exists today.
- docs/NETWORKING.md -- Windows Firewall rules for multi-machine setup.
