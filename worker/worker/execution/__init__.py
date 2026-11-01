"""Pipeline-stage execution engine: receives an incoming activation
tensor, runs it through this worker's assigned layers, and returns the
result to the coordinator, which sends it on to the next stage's worker.

Phase 6: the StageExecution gRPC service (service.py, server.py) for the
built-in test model. It imports the stage implementation from
model-partitioner/ (and therefore torch + transformers) only when the
service is used.
"""
