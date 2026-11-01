"""End-to-end planning on the real GPT-2 spec with synthetic worker
records: CPU-only, GPU, heterogeneous, unequal memory, missing data,
infeasible requests, labelling of simulated workers, determinism."""

from __future__ import annotations

import datetime as dt
import json
import random

import pytest

from conftest import gpu_worker, make_worker, replace
from partitioner import create_plan, validate_plan
from partitioner.api import PlanningError
from partitioner.cost import CostModel, WorkerOverride
from partitioner.memory import MemoryPolicy, Precision, Workload
from partitioner.workers import GIB

FIXED_NOW = dt.datetime(2026, 10, 31, tzinfo=dt.timezone.utc)


def plan_for(spec, workers, **kw):
    kw.setdefault("now", FIXED_NOW)
    kw.setdefault("cost_model", CostModel(throughput_preference="estimated"))
    return create_plan(spec, workers, **kw)


def blocks_of(stage):
    lr = stage["layer_range"]
    return 0 if lr is None else lr["last_block"] - lr["first_block"] + 1


def test_single_worker_gets_the_whole_model(gpt2_spec):
    plan = plan_for(gpt2_spec, [make_worker("only")])
    assert plan["feasible"]
    (st,) = plan["stages"]
    assert st["unit_range"] == [0, 15] and st["layer_range"] == {"first_block": 0, "last_block": 11}
    assert st["is_bottleneck"] and st["memory_fits"]
    assert plan["objective"]["bottleneck_seconds"] == st["estimated_seconds"]


def test_heterogeneous_speed_gives_faster_worker_more_layers(gpt2_spec):
    fast = make_worker("fast", cores=16, mhz=4000)
    slow = make_worker("slow", cores=4, mhz=2000)
    plan = plan_for(gpt2_spec, [fast, slow], allow_unused_workers=False)
    by = {s["worker_id"]: s for s in plan["stages"]}
    assert blocks_of(by["fast"]) > blocks_of(by["slow"])
    times = [s["estimated_seconds"] for s in plan["stages"]]
    assert max(times) == plan["objective"]["bottleneck_seconds"]


def test_equal_workers_split_roughly_evenly_by_cost_not_count(gpt2_spec):
    plan = plan_for(gpt2_spec, [make_worker("a"), make_worker("b")], allow_unused_workers=False)
    t = [s["estimated_seconds"] for s in plan["stages"]]
    assert max(t) / min(t) < 1.6  # balanced by estimated cost
    # Not an equal UNIT count: the lm_head (a 50257x768 matmul) weighs ~5 blocks.
    assert plan["stages"][0]["unit_range"] != [0, 8]


def test_memory_limited_worker_gets_less(gpt2_spec):
    # Fast-but-low-memory worker vs. slow worker with lots of memory.
    # 1.4 GiB: enough for the embedding end of the chain plus some
    # blocks, not the whole model (see the next test for why the
    # amount matters).
    small_mem = make_worker("small-mem", ram_usable_gib=1.4, cores=32, mhz=4000)
    big_mem = make_worker("big-mem", ram_usable_gib=16, cores=4, mhz=2000)
    plan = plan_for(gpt2_spec, [small_mem, big_mem])
    assert plan["feasible"]
    for st in plan["stages"]:
        assert st["memory"]["total_bytes"] <= st["available_memory"]["device_capacity_bytes"]
    by = {s["worker_id"]: s for s in plan["stages"]}
    assert "small-mem" in by  # it is faster, so it is used...
    assert by["small-mem"]["unit_range"] != [0, 15]  # ...but can't hold everything


def test_end_stages_are_heavier_than_middle_stages(gpt2_spec):
    """With 2 workers every stage is an END stage (embeddings, or the LM
    head with its logits and tied-weight copy). A 0.95 GiB worker fits
    neither end, so it can't help -- but with 3 workers it fits in the
    middle and is used there."""
    tiny = make_worker("tiny", ram_usable_gib=0.95, cores=32, mhz=4000)
    two = plan_for(gpt2_spec, [tiny, make_worker("big", ram_usable_gib=16, cores=4, mhz=2000)])
    assert [s["worker_id"] for s in two["stages"]] == ["big"]
    assert "not needed" in next(u["reason"] for u in two["unused_workers"] if u["worker_id"] == "tiny")

    three = plan_for(gpt2_spec, [tiny, make_worker("big1", ram_usable_gib=16, cores=4, mhz=2000),
                                 make_worker("big2", ram_usable_gib=16, cores=4, mhz=2000)])
    ids = [s["worker_id"] for s in three["stages"]]
    assert ids.index("tiny") not in (0, len(ids) - 1)  # middle only
    st = next(s for s in three["stages"] if s["worker_id"] == "tiny")
    assert st["layer_range"] is not None and "embeddings" not in st["units"] and "lm_head" not in st["units"]


def test_unused_reason_distinguishes_too_small_from_not_needed(gpt2_spec):
    # "spare" is so slow that even the cheapest unit would cost more on
    # it than big's whole stage (a merely slow worker IS used: taking
    # e.g. final_norm still lowers the bottleneck slightly).
    plan = plan_for(gpt2_spec, [make_worker("big", ram_usable_gib=16, cores=64),
                                make_worker("crumbs", ram_usable_gib=0.1),
                                make_worker("spare", ram_usable_gib=16)],
                    overrides={"spare": WorkerOverride(cpu_gflops=1e-6)})
    reasons = {u["worker_id"]: u["reason"] for u in plan["unused_workers"]}
    assert reasons["crumbs"].startswith("too little memory for any stage")
    assert "not needed" in reasons["spare"]
    assert all(u["ram_usable_bytes"] is not None for u in plan["unused_workers"])


def test_cpu_only_worker_never_gets_gpu_memory(gpt2_spec):
    plan = plan_for(gpt2_spec, [make_worker("cpu-box")])
    st = plan["stages"][0]
    assert st["execution"]["device"] == "cpu" and st["execution"]["gpu_id"] is None
    assert st["available_memory"]["device_capacity_bytes"] == st["available_memory"]["host_capacity_bytes"]


def test_gpu_worker_uses_vram_when_throughput_known(gpt2_spec):
    g = gpu_worker("gpu-box", vram_usable_gib=6)
    plan = plan_for(gpt2_spec, [g], overrides={"gpu-box": WorkerOverride(gpu_gflops=5000)})
    st = plan["stages"][0]
    assert st["execution"]["device"] == "gpu" and st["execution"]["backend"] == "cuda"
    assert st["available_memory"]["device_capacity_bytes"] == int(6 * GIB * 0.9)
    # Runtime overhead is host RAM, not VRAM.
    assert st["memory"]["host_bytes"] >= 700 * 2**20


def test_gpu_without_known_throughput_falls_back_to_cpu(gpt2_spec):
    plan = plan_for(gpt2_spec, [gpu_worker("gpu-box")])
    assert plan["stages"][0]["execution"]["device"] == "cpu"
    assert any("no GPU throughput is known" in w for w in plan["warnings"])


def test_gpu_vram_limit_respected(gpt2_spec):
    tiny = gpu_worker("tiny-vram", vram_usable_gib=0.3)
    cpu = make_worker("cpu", ram_usable_gib=8)
    plan = plan_for(gpt2_spec, [tiny, cpu], overrides={"tiny-vram": WorkerOverride(gpu_gflops=10000)})
    for st in plan["stages"]:
        if st["execution"]["device"] == "gpu":
            assert st["memory"]["device_bytes"] <= st["available_memory"]["device_capacity_bytes"]


@pytest.mark.parametrize(
    "worker, reason",
    [
        (make_worker("no-ram", ram_usable_gib=None), "available RAM is unknown"),
        (make_worker("offline", online=False), "offline"),
        (make_worker("pending", backend_probe_pending=True), "probe has not finished"),
        (make_worker("no-torch", execution_backends=()), "no usable CPU inference backend"),
        (make_worker("no-clock", cores=0, mhz=None), "CPU throughput unknown"),
        (make_worker("unknown-src", profile_source="UNKNOWN"), "neither REAL nor SIMULATED"),
    ],
)
def test_incomplete_or_unusable_workers_excluded_with_reason(gpt2_spec, worker, reason):
    plan = plan_for(gpt2_spec, [worker, make_worker("ok")])
    assert [s["worker_id"] for s in plan["stages"]] == ["ok"]
    (u,) = [u for u in plan["unused_workers"] if u["worker_id"] == worker.worker_id]
    assert reason in u["reason"]


def test_insufficient_memory_is_infeasible_and_explained(gpt2_spec):
    workers = [make_worker(f"w{i}", ram_usable_gib=0.6) for i in range(4)]
    plan = plan_for(gpt2_spec, workers)
    assert plan["feasible"] is False and plan["stages"] == []
    reasons = " ".join(plan["infeasibility"]["reasons"])
    assert "whole model on one worker needs" in reasons and "runtime overhead" in reasons
    validate_plan(plan, gpt2_spec)  # an infeasible plan is still a valid document


def test_no_workers_is_infeasible(gpt2_spec):
    plan = plan_for(gpt2_spec, [make_worker("x", online=False)])
    assert not plan["feasible"] and "No usable workers." in plan["infeasibility"]["reasons"]


def test_simulated_workers_are_labelled_and_can_be_excluded(gpt2_spec):
    real = make_worker("real-1", profile_source="REAL", simulated_profile_name="")
    sim = make_worker("sim-1", cores=32)
    plan = plan_for(gpt2_spec, [real, sim])
    assert plan["includes_simulated_workers"] is True
    assert any("SIMULATED - NOT PHYSICALLY VERIFIED" in w for w in plan["warnings"])
    assert {s["worker_id"]: s["profile_source"] for s in plan["stages"]}.get("sim-1") == "SIMULATED"

    only_real = plan_for(gpt2_spec, [real, sim], allow_simulated=False)
    assert [s["worker_id"] for s in only_real["stages"]] == ["real-1"]
    assert only_real["includes_simulated_workers"] is False


def test_shared_host_splits_ram(gpt2_spec):
    a = make_worker("a", profile_source="REAL", hostname="YakobsPC", ram_usable_gib=4)
    b = make_worker("b", profile_source="REAL", hostname="YakobsPC", ram_usable_gib=4)
    plan = plan_for(gpt2_spec, [a, b], allow_unused_workers=False)
    for st in plan["stages"]:
        assert st["execution"]["ram_share"] == 0.5
        assert st["available_memory"]["device_capacity_bytes"] == int(4 * GIB * 0.5 * 0.9)
    assert any("NOT independent computers" in w for w in plan["warnings"])


def test_tied_lm_head_copy_charged_to_last_stage(gpt2_spec):
    plan = plan_for(gpt2_spec, [make_worker("a"), make_worker("b")], allow_unused_workers=False)
    last = plan["stages"][-1]
    assert any(i["name"] == "tied_weight_copy" for i in last["memory"]["items"])
    assert not any(i["name"] == "tied_weight_copy" for i in plan["stages"][0]["memory"]["items"])


def test_require_all_workers(gpt2_spec):
    workers = [make_worker("big", cores=64), make_worker("tiny", cores=1, mhz=500)]
    relaxed = plan_for(gpt2_spec, workers)
    strict = plan_for(gpt2_spec, workers, allow_unused_workers=False)
    assert {s["worker_id"] for s in strict["stages"]} == {"big", "tiny"}
    assert relaxed["objective"]["bottleneck_seconds"] <= strict["objective"]["bottleneck_seconds"]


def test_deterministic_and_independent_of_input_order(gpt2_spec):
    rng = random.Random(42)
    workers = [make_worker(f"w{i}", cores=rng.randint(2, 16), mhz=rng.choice([2000, 3000, 4000]),
                           ram_usable_gib=rng.choice([1.0, 2.0, 8.0])) for i in range(5)]
    first = plan_for(gpt2_spec, workers)
    for _ in range(3):
        shuffled = workers[:]
        rng.shuffle(shuffled)
        again = plan_for(gpt2_spec, shuffled)
        assert again["inputs_fingerprint"] == first["inputs_fingerprint"]
        assert again["stages"] == first["stages"]


def test_timestamp_does_not_change_fingerprint(gpt2_spec):
    w = [make_worker("a")]
    p1 = plan_for(gpt2_spec, w, now=FIXED_NOW)
    p2 = plan_for(gpt2_spec, w, now=FIXED_NOW + dt.timedelta(days=1))
    assert p1["created_at"] != p2["created_at"] and p1["inputs_fingerprint"] == p2["inputs_fingerprint"]


def test_randomized_plans_always_valid(gpt2_spec):
    """Property: for random heterogeneous clusters, every feasible plan
    covers all units once, in order, within memory -- and validates."""
    rng = random.Random(2026)
    for trial in range(40):
        n = rng.randint(1, 5)
        workers = []
        for i in range(n):
            if rng.random() < 0.3:
                w = gpu_worker(f"g{i}", vram_usable_gib=rng.choice([0.2, 1.0, 4.0]))
            else:
                w = make_worker(f"c{i}", cores=rng.randint(1, 24), mhz=rng.choice([1500, 3000, 4500]),
                                ram_usable_gib=rng.choice([0.5, 0.9, 1.5, 4.0]))
            workers.append(w)
        overrides = {w.worker_id: WorkerOverride(gpu_gflops=rng.choice([500, 5000])) for w in workers if w.gpus}
        precision = Precision(rng.choice(["fp32", "fp16", "bf16"]), rng.choice(["none", "none", "int8", "int4"]))
        workload = Workload(batch_size=rng.choice([1, 2]), context_length=rng.choice([128, 512, 1024]))
        plan = plan_for(gpt2_spec, workers, precision=precision, workload=workload, overrides=overrides)
        if not plan["feasible"]:
            assert plan["infeasibility"]["reasons"], f"trial {trial}"
            continue
        covered = [u for s in plan["stages"] for u in s["units"]]
        assert covered == [u.name for u in gpt2_spec.units], f"trial {trial}"
        assert all(s["memory_fits"] for s in plan["stages"]), f"trial {trial}"
        validate_plan(plan, gpt2_spec, {w.worker_id: w for w in workers})


def test_plan_schema_fields_and_json_roundtrip(gpt2_spec, tmp_path):
    from partitioner import load_plan, save_plan

    plan = plan_for(gpt2_spec, [make_worker("a"), make_worker("b", cores=4)])
    for key in ("partition_plan_schema_version", "created_at", "planner", "model", "precision", "workload",
                "memory_policy", "cost_model", "stages", "unused_workers", "objective", "feasible", "warnings",
                "assumptions", "includes_simulated_workers", "executable", "inputs_fingerprint"):
        assert key in plan, key
    assert plan["partition_plan_schema_version"] == 1
    assert plan["executable"] is False and "Phase 4 produces plans only" in plan["executable_reason"]
    st = plan["stages"][0]
    for key in ("worker_id", "profile_source", "unit_range", "layer_range", "components", "memory",
                "available_memory", "estimated_seconds", "is_bottleneck", "explanation"):
        assert key in st, key
    path = tmp_path / "plan.json"
    save_plan(plan, str(path))
    loaded = load_plan(str(path))
    assert loaded == json.loads(json.dumps(plan))
    validate_plan(loaded, gpt2_spec)


def test_explanations_cover_memory_time_and_boundaries(gpt2_spec):
    plan = plan_for(gpt2_spec, [make_worker("a"), make_worker("b")], allow_unused_workers=False)
    text = " ".join(plan["stages"][0]["explanation"])
    assert "Memory: needs" in text and "per step" in text and "boundary" in text


def test_link_bandwidth_adds_transfer_cost(gpt2_spec):
    workers = [make_worker("a"), make_worker("b")]
    without = plan_for(gpt2_spec, workers, allow_unused_workers=False)
    slow_link = plan_for(gpt2_spec, workers, allow_unused_workers=False,
                         cost_model=CostModel(throughput_preference="estimated", link_bandwidth_bytes_per_s=1e5))
    assert slow_link["objective"]["bottleneck_seconds"] > without["objective"]["bottleneck_seconds"]
    assert not any("NOT modelled" in w for w in slow_link["warnings"])
    first, last = slow_link["stages"][0], slow_link["stages"][-1]
    # 768 hidden x 4 B per decode step over a 1e5 B/s link.
    assert first["estimated_transfer_seconds"] == pytest.approx(768 * 4 / 1e5)
    assert first["estimated_seconds"] == pytest.approx(first["estimated_compute_seconds"] + first["estimated_transfer_seconds"])
    assert last["estimated_transfer_seconds"] == 0  # the last stage sends nothing
    assert "sending its output" in " ".join(first["explanation"])


def test_bad_inputs_rejected(gpt2_spec):
    with pytest.raises(PlanningError, match="duplicate worker ids"):
        plan_for(gpt2_spec, [make_worker("a"), make_worker("a")])
    with pytest.raises(PlanningError, match="unknown workers"):
        plan_for(gpt2_spec, [make_worker("a")], overrides={"ghost": WorkerOverride(cpu_gflops=1)})


def test_utilization_reduces_throughput(gpt2_spec):
    idle = plan_for(gpt2_spec, [make_worker("a", cpu_utilization_percent=0.0)])
    busy = plan_for(gpt2_spec, [make_worker("a", cpu_utilization_percent=75.0)])
    assert busy["objective"]["bottleneck_seconds"] == pytest.approx(4 * idle["objective"]["bottleneck_seconds"])


def test_synthetic_small_model_plans(small_spec):
    plan = plan_for(small_spec, [make_worker("a"), make_worker("b")], allow_unused_workers=False)
    assert [u for s in plan["stages"] for u in s["units"]] == [u.name for u in small_spec.units]
    assert len(small_spec.units) == 7
