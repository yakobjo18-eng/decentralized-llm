from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from partitioner.inspection import ModelInspectionError, inspect_model
from partitioner.workers import GIB, GPUResource, WorkerResources

REPO = Path(__file__).resolve().parents[2]
PROFILES = REPO / "worker" / "profiles" / "simulated"


@pytest.fixture(scope="session")
def gpt2_spec():
    """The REAL GPT-2 (config.json from the local HF cache or the Hub;
    meta device; checkpoint header checked). Skips -- never fakes -- if
    the config can't be obtained."""
    try:
        return inspect_model("gpt2", check_checkpoint=True)
    except ModelInspectionError as exc:
        pytest.skip(f"GPT-2 config unavailable (offline and not cached): {exc}")


@pytest.fixture(scope="session")
def small_spec():
    """A synthetic 4-layer GPT-2 config, fully offline -- proves nothing
    downstream hardcodes GPT-2 small's 12 layers."""
    from transformers import GPT2Config

    cfg = GPT2Config(n_layer=4, n_embd=64, n_head=4, vocab_size=1000, n_positions=128)
    return inspect_model(config=cfg, check_checkpoint=False)


def make_worker(worker_id: str, ram_usable_gib: float = 8.0, cores: int = 8, mhz: float = 3000.0, **kw) -> WorkerResources:
    usable = None if ram_usable_gib is None else int(ram_usable_gib * GIB)
    defaults = dict(
        worker_id=worker_id,
        profile_source="SIMULATED",
        online=True,
        ram_total_bytes=int(64 * GIB),
        ram_available_bytes=None if usable is None else usable + GIB,
        ram_safety_margin_bytes=GIB,
        ram_usable_bytes=usable,
        cpu_physical_cores=cores,
        cpu_logical_cores=cores * 2,
        cpu_utilization_percent=0.0,
        cpu_frequency_max_mhz=mhz,
        execution_backends=("cpu",),
        hostname=f"host-{worker_id}",
        simulated_profile_name=f"test-{worker_id}",
    )
    defaults.update(kw)
    return WorkerResources(**defaults)


def gpu_worker(worker_id: str, vram_usable_gib: float = 6.0, **kw) -> WorkerResources:
    g = GPUResource("sim:gpu:0", "Test GPU", "NVIDIA", "dedicated", True, int(8 * GIB), int(GIB), int(vram_usable_gib * GIB), 0.0)
    return make_worker(worker_id, execution_backends=("cpu", "cuda"), gpus=(g,), **kw)


@pytest.fixture
def worker_factory():
    return make_worker


@pytest.fixture
def gpu_worker_factory():
    return gpu_worker


def replace(obj, **kw):
    return dataclasses.replace(obj, **kw)


# ---- Phase 5 execution helpers ------------------------------------------


@pytest.fixture(scope="session")
def tiny_spec():
    from partitioner.execution import tiny_gpt2

    return tiny_gpt2.spec()


def real_local_worker(worker_id: str = "local-1", **kw) -> WorkerResources:
    """A REAL-labelled record for THIS host. Synthetic test input (fixed
    8 GiB usable), not a measurement: it exercises the real-mode code
    path without depending on how much RAM the machine has free."""
    import socket

    kw.setdefault("hostname", socket.gethostname())
    return make_worker(worker_id, profile_source="REAL", simulated_profile_name="",
                       origin="synthetic test record (not a measurement)", **kw)


def tiny_plan(spec, workers, **kw) -> dict:
    import datetime as dt

    from partitioner import create_plan
    from partitioner.cost import CostModel

    kw.setdefault("now", dt.datetime(2026, 11, 1, tzinfo=dt.timezone.utc))
    kw.setdefault("cost_model", CostModel(throughput_preference="estimated"))
    return create_plan(spec, workers, **kw)


def refingerprint(plan: dict) -> dict:
    """For tests that edit a plan to reach checks BEHIND the fingerprint
    check: re-seal it so only the edited content is wrong."""
    from partitioner import plan as plan_mod

    plan["inputs_fingerprint"] = plan_mod.fingerprint(plan)
    return plan
