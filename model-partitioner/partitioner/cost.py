"""Execution-cost model: FLOPs per unit, and per-worker throughput.

Stage time = FLOPs(units on the stage) / effective throughput
             (+ optional transfer time of the stage's output).

FLOPs are CALCULATED from the real parameter shapes (2 x tokens x
weights for every matmul, plus attention's QK^T and AV); elementwise
work (norms, GELU, softmax, residuals) is ignored. The resulting
*seconds* are always ESTIMATED: throughput is either a noisy
micro-benchmark (Phase 3's 256x256 matmul test, which spread ~10x
between runs on Computer 1), a configured value, or derived from core
count x clock x an assumed FLOPs-per-cycle. Decode on CPUs is often
memory-bandwidth bound, which this compute-only model doesn't capture
(docs/MODEL_PARTITIONING.md, "Limitations").

The optimizer only needs RELATIVE stage costs to be right; absolute
seconds are reported but are not predictions.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple

from partitioner import model_spec as ms
from partitioner.memory import DTYPE_BYTES, MemoryPolicy, Precision, Workload
from partitioner.provenance import Provenance
from partitioner.workers import WorkerResources


@dataclasses.dataclass(frozen=True)
class CostModel:
    # Used only when no measured or configured throughput exists. 8
    # fp32 FLOPs/cycle/core is a conservative fraction (~25%) of an
    # AVX2 core's 32-FLOP FMA peak -- an assumption, not a measurement.
    cpu_flops_per_cycle_per_core: float = 8.0
    # "measured": prefer the Phase 3 micro-benchmark when present.
    # "estimated": always use the core-count formula (stabler, since the
    # micro-benchmark is noisy).
    throughput_preference: str = "measured"
    # Scale throughput by the idle fraction (1 - utilization).
    account_for_utilization: bool = True
    min_available_compute_fraction: float = 0.10
    # Inter-stage link. None = transfer time not modelled (warned).
    link_bandwidth_bytes_per_s: Optional[float] = None
    link_latency_s: float = 0.0

    def validate(self) -> None:
        if self.throughput_preference not in ("measured", "estimated"):
            raise ValueError("throughput_preference must be 'measured' or 'estimated'")
        if self.cpu_flops_per_cycle_per_core <= 0:
            raise ValueError("cpu_flops_per_cycle_per_core must be positive")
        if not 0 < self.min_available_compute_fraction <= 1:
            raise ValueError("min_available_compute_fraction must be in (0, 1]")
        if self.link_bandwidth_bytes_per_s is not None and self.link_bandwidth_bytes_per_s <= 0:
            raise ValueError("link_bandwidth_bytes_per_s must be positive")


@dataclasses.dataclass(frozen=True)
class WorkerOverride:
    """Per-worker values supplied by the operator (e.g. a GPU's
    throughput, which no Phase 3 measurement provides yet)."""

    cpu_gflops: Optional[float] = None
    gpu_gflops: Optional[float] = None
    force_cpu: bool = False


@dataclasses.dataclass(frozen=True)
class ExecutionTarget:
    worker: WorkerResources
    device: str  # "cpu" | "gpu"
    backend: str  # "cpu" | "cuda" | ...
    gpu_id: Optional[str]
    flops_per_s: float
    throughput_provenance: Provenance
    throughput_basis: str
    utilization_factor: float
    device_capacity_bytes: int  # pool that holds weights/KV/activations
    host_capacity_bytes: int  # RAM for runtime + load buffers (== device pool on CPU)
    capacity_basis: str
    ram_share: float = 1.0

    def to_dict(self) -> dict:
        return {
            "device": self.device,
            "backend": self.backend,
            "gpu_id": self.gpu_id,
            "effective_gflops": round(self.flops_per_s / 1e9, 3),
            "throughput_provenance": self.throughput_provenance.value,
            "throughput_basis": self.throughput_basis,
            "utilization_factor": round(self.utilization_factor, 4),
            "device_capacity_bytes": self.device_capacity_bytes,
            "host_capacity_bytes": self.host_capacity_bytes,
            "capacity_basis": self.capacity_basis,
            "ram_share": self.ram_share,
        }


def _util_factor(util: Optional[float], cm: CostModel) -> float:
    if not cm.account_for_utilization or util is None:
        return 1.0
    return max(cm.min_available_compute_fraction, 1.0 - util / 100.0)


def resolve_target(
    worker: WorkerResources,
    override: WorkerOverride,
    cost_model: CostModel,
    policy: MemoryPolicy,
    ram_share: float = 1.0,
) -> Tuple[Optional[ExecutionTarget], List[str], str]:
    """Decides how a worker would execute (CPU or a usable GPU) and with
    what capacity. Returns (target, warnings, reason-if-unusable).
    Missing information makes a worker unusable, never optimistic."""
    warnings: List[str] = []
    if not worker.online:
        return None, warnings, "worker is offline"
    if worker.backend_probe_pending:
        return None, warnings, "worker's inference backend probe has not finished; its capabilities are unknown"
    if worker.ram_usable_bytes is None:
        return None, warnings, "available RAM is unknown, so no memory can be assumed free"
    keep = 1.0 - policy.extra_safety_fraction
    host_cap = int(worker.ram_usable_bytes * ram_share * keep)
    share_note = f" x RAM share {ram_share:.3f}" if ram_share != 1.0 else ""

    # GPU path: a compute-usable GPU with known usable VRAM AND a
    # known throughput. Otherwise plan on the CPU (conservative).
    if not override.force_cpu:
        for g in worker.gpus:
            if not g.compute_usable:
                continue
            backend = next((b for b in worker.execution_backends if b != "cpu"), None)
            if g.vram_usable_bytes is None:
                warnings.append(f"{worker.worker_id}: GPU {g.name} is compute-usable but its usable VRAM is unknown; planned on CPU")
                continue
            gflops = override.gpu_gflops
            prov, basis = Provenance.ESTIMATED, "operator-configured GPU throughput"
            if gflops is None and backend and worker.gflops.get(backend) and cost_model.throughput_preference == "measured":
                gflops = worker.gflops[backend]
                prov, basis = Provenance.ESTIMATED, f"Phase 3 {backend} matmul micro-benchmark (MEASURED, noisy)"
            if gflops is None:
                warnings.append(
                    f"{worker.worker_id}: GPU {g.name} is compute-usable but no GPU throughput is known "
                    "(pass a gpu_gflops override); planned on CPU instead"
                )
                continue
            uf = _util_factor(g.utilization_percent, cost_model)
            return ExecutionTarget(
                worker, "gpu", backend or "gpu", g.gpu_id, gflops * 1e9 * uf, prov, basis, uf,
                int(g.vram_usable_bytes * keep), host_cap,
                f"VRAM usable {g.vram_usable_bytes} B x (1 - extra safety {policy.extra_safety_fraction}); "
                f"host RAM usable {worker.ram_usable_bytes} B{share_note} x (1 - {policy.extra_safety_fraction})",
                ram_share,
            ), warnings, ""

    if "cpu" not in worker.execution_backends:
        return None, warnings, (
            f"no usable CPU inference backend (execution_backends={list(worker.execution_backends)}); "
            "e.g. PyTorch not installed"
        )
    gflops, prov, basis = override.cpu_gflops, Provenance.ESTIMATED, "operator-configured CPU throughput"
    if gflops is None and cost_model.throughput_preference == "measured" and worker.gflops.get("cpu"):
        gflops = worker.gflops["cpu"]
        basis = "Phase 3 cpu matmul micro-benchmark (MEASURED once; ~10x run-to-run spread observed on Computer 1)"
    if gflops is None:
        cores, mhz = worker.cpu_physical_cores, worker.cpu_frequency_max_mhz
        if not cores or not mhz:
            return None, warnings, "CPU throughput unknown: no measurement, no override, and core count or clock unreported"
        gflops = cores * mhz * 1e6 * cost_model.cpu_flops_per_cycle_per_core / 1e9
        basis = (
            f"{cores} physical cores x {mhz:.0f} MHz x {cost_model.cpu_flops_per_cycle_per_core} FLOPs/cycle (assumed)"
        )
    uf = _util_factor(worker.cpu_utilization_percent, cost_model)
    return ExecutionTarget(
        worker, "cpu", "cpu", None, gflops * 1e9 * uf, prov, basis, uf, host_cap, host_cap,
        f"RAM usable {worker.ram_usable_bytes} B{share_note} x (1 - extra safety {policy.extra_safety_fraction})",
        ram_share,
    ), warnings, ""


def unit_flops(spec: ms.ModelSpec, unit: ms.Unit, workload: Workload) -> float:
    """FLOPs for one forward pass of `unit` under the workload's cost
    mode. decode: one new token per sequence attending over the full
    context. prefill: the whole prompt (causal masking ignored, i.e. a
    conservative full T x T attention)."""
    B = workload.batch_size
    if workload.cost_mode == "decode":
        T, L = 1, workload.context_length
    else:
        T, L = workload.prompt_length, workload.prompt_length
    tokens = B * T
    h = spec.hidden_size
    if unit.kind == ms.TRANSFORMER_BLOCK:
        linear = sum(p.numel for c in spec.unit_components(unit) for p in c.params if p.kind == ms.LINEAR_WEIGHT)
        attn = 4 * B * T * L * spec.attention.n_heads * spec.attention.head_dim
        return 2.0 * tokens * linear + attn
    if unit.kind == ms.LM_HEAD:
        positions = T if (workload.cost_mode == "prefill" and workload.logits_positions == "all") else 1
        return 2.0 * B * positions * spec.vocab_size * h
    # embeddings gather / final norm: elementwise, small but non-zero
    return 5.0 * tokens * h


def transfer_seconds(spec: ms.ModelSpec, workload: Workload, precision: Precision, cost_model: CostModel) -> Optional[float]:
    """Time to send one stage's output hidden state to the next stage,
    or None if the link isn't modelled."""
    if cost_model.link_bandwidth_bytes_per_s is None:
        return None
    T = 1 if workload.cost_mode == "decode" else workload.prompt_length
    nbytes = workload.batch_size * T * spec.hidden_size * DTYPE_BYTES[precision.act]
    return cost_model.link_latency_s + nbytes / cost_model.link_bandwidth_bytes_per_s


def unit_flops_table(spec: ms.ModelSpec, workload: Workload) -> Dict[int, float]:
    return {u.index: unit_flops(spec, u, workload) for u in spec.units}
