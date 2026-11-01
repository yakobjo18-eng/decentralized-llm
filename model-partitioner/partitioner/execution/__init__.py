"""Phase 5: minimal, local, real inference for a validated plan.

Executes only the built-in test model builtin:tiny-gpt2-v1. Local
modes (runner.py) run every stage in one process. Phase 6 adds remote
execution (remote.py, wire.py): each stage runs on its assigned worker's
StageExecution service, with activations sent over gRPC (unauthenticated).
"""

from partitioner.execution.interface import (
    STAGE_INTERFACE_VERSION,
    Activation,
    ActivationMeta,
    ActivationTransport,
    InProcessTransport,
    InterfaceError,
    StageExecutor,
    StageSpec,
)
from partitioner.execution.limits import ExecutionLimits
from partitioner.execution.loader import LOCAL_EMULATION, REAL_MODE, ExecutionRejected, load_for_execution
from partitioner.execution.models import EXECUTABLE_MODEL_IDS, TINY_GPT2_ID, get_executable_model
from partitioner.execution.runner import REFERENCE_ATOL, BoundaryError, execute_plan

__all__ = [
    "Activation", "ActivationMeta", "ActivationTransport", "BoundaryError", "EXECUTABLE_MODEL_IDS", "ExecutionLimits",
    "ExecutionRejected", "InProcessTransport", "InterfaceError", "LOCAL_EMULATION", "REAL_MODE", "REFERENCE_ATOL",
    "STAGE_INTERFACE_VERSION", "StageExecutor", "StageSpec", "TINY_GPT2_ID", "execute_plan", "get_executable_model",
    "load_for_execution",
]
