# Cross-component integration tests

Not yet implemented as automated tests. Phase 1's equivalent is the
manual-run `scripts/smoke_test_phase1.ps1`, which drives the real
orchestrator binary and a real worker process over loopback gRPC/REST
and checks the result (see docs/RUNNING_PHASE1.md).

The final end-to-end test described in the master prompt (three
physical machines, real model partitioning, real distributed
inference, recorded performance metrics) lands in Phase 10, once
Phases 2-9 exist to integrate.
