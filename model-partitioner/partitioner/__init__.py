"""Hardware-aware Transformer model partitioning engine (Phase 4).

Inspects a real Hugging Face model (configuration + meta device, no
weights), estimates per-stage memory, and assigns contiguous layer
ranges to heterogeneous workers so the slowest stage is as fast as
possible. Phase 5 adds partitioner.execution: real, single-process
inference for plans of the built-in test model builtin:tiny-gpt2-v1.
No tensor transport or distributed execution exists yet.
"""

from partitioner.adapters import AdapterError, get_adapter
from partitioner.api import PlanningError, create_plan
from partitioner.cost import CostModel, WorkerOverride
from partitioner.inspection import ModelInspectionError, inspect_model, load_config
from partitioner.memory import EstimationError, MemoryEstimator, MemoryPolicy, Precision, Workload
from partitioner.model_spec import ModelSpec
from partitioner.plan import PLAN_SCHEMA_VERSION, load as load_plan, save as save_plan
from partitioner.validate import PlanValidationError, validate_plan
from partitioner.workers import (
    Phase3Margins,
    WorkerResources,
    from_local_machine,
    from_orchestrator,
    from_resources_json,
    from_simulated_profile,
)

__all__ = [
    "AdapterError", "CostModel", "EstimationError", "MemoryEstimator", "MemoryPolicy", "ModelInspectionError",
    "ModelSpec", "PLAN_SCHEMA_VERSION", "Phase3Margins", "PlanValidationError", "PlanningError", "Precision",
    "WorkerOverride", "WorkerResources", "Workload", "create_plan", "from_local_machine", "from_orchestrator",
    "from_resources_json", "from_simulated_profile", "get_adapter", "inspect_model", "load_config", "load_plan",
    "save_plan", "validate_plan",
]
