"""The DP optimizer against an exhaustive oracle on randomized
instances (seeded, so failures are reproducible), plus targeted
properties: determinism, feasibility, bottleneck vs. equal splits."""

from __future__ import annotations

import random

import pytest

from partitioner.optimizer import brute_force, check_ranges, solve


def random_instance(rng: random.Random, U: int, m: int, p_infeasible: float):
    """Per-worker per-unit costs, plus per-worker memory caps on range
    length (models 'too many layers don't fit')."""
    unit_cost = [[rng.uniform(0.1, 10.0) for _ in range(U)] for _ in range(m)]
    cap = [rng.randint(1, U) if rng.random() < p_infeasible else U for _ in range(m)]

    def cost(k, i, j):
        if j - i > cap[k]:
            return None
        return sum(unit_cost[k][i:j])

    return cost


def test_dp_matches_brute_force_on_1500_random_instances():
    infeasible_seen = 0
    for seed in range(1500):
        rng = random.Random(seed)
        U, m = rng.randint(1, 7), rng.randint(1, 4)
        allow_empty = rng.random() < 0.7
        cost = random_instance(rng, U, m, p_infeasible=0.5)
        sol = solve(cost, U, m, allow_empty)
        oracle = brute_force(cost, U, m, allow_empty)
        if oracle is None:
            assert sol is None, f"seed {seed}: DP found a plan the oracle says is infeasible"
            infeasible_seen += 1
            continue
        assert sol is not None, f"seed {seed}: DP missed a feasible plan"
        assert sol.bottleneck == pytest.approx(oracle, rel=1e-12), f"seed {seed}"
        assert not check_ranges(sol.ranges, U), f"seed {seed}"
        for k, (s, e) in enumerate(sol.ranges):
            assert sol.stage_costs[k] == (0.0 if s == e else cost(k, s, e)), f"seed {seed}"
            assert allow_empty or e > s, f"seed {seed}"
    assert infeasible_seen > 50  # the generator really exercises infeasibility


def test_secondary_objective_minimizes_total_among_optimal():
    """Among all bottleneck-optimal partitions, the DP's total cost is
    minimal (checked exhaustively on 200 random instances)."""
    for seed in range(200):
        _check_total_optimal(seed)


def _check_total_optimal(seed: int) -> None:
    rng = random.Random(1000 + seed)
    U, m = rng.randint(2, 6), rng.randint(2, 3)
    cost = random_instance(rng, U, m, p_infeasible=0.3)
    sol = solve(cost, U, m, True)
    if sol is None:
        return
    best_total = None

    def rec(k, start, worst, total):
        nonlocal best_total
        ends = [U] if k == m - 1 else range(start, U + 1)
        for end in ends:
            c = 0.0 if start == end else cost(k, start, end)
            if c is None:
                continue
            w, t = max(worst, c), total + c
            if k == m - 1:
                if w <= sol.bottleneck * (1 + 1e-12) and (best_total is None or t < best_total):
                    best_total = t
            else:
                rec(k + 1, end, w, t)

    rec(0, 0, 0.0, 0.0)
    assert sol.total == pytest.approx(best_total, rel=1e-9), f"seed {seed}"


def test_deterministic():
    cost = random_instance(random.Random(7), 12, 3, 0.2)
    first = solve(cost, 12, 3)
    for _ in range(5):
        assert solve(cost, 12, 3) == first


def test_bottleneck_not_equal_layer_counts():
    """Worker 0 is 3x faster: the optimum gives it ~3x the units, not an
    equal split."""
    speed = [3.0, 1.0]
    cost = lambda k, i, j: (j - i) / speed[k]  # noqa: E731
    sol = solve(cost, 12, 2, allow_empty=False)
    assert sol.ranges == ((0, 9), (9, 12))
    assert sol.bottleneck == pytest.approx(3.0)


def test_single_worker_gets_everything():
    sol = solve(lambda k, i, j: float(j - i), 15, 1)
    assert sol.ranges == ((0, 15),) and sol.bottleneck == 15.0


def test_memory_cap_forces_split():
    # Worker 0 can hold at most 4 units even though it is much faster.
    cost = lambda k, i, j: None if (k == 0 and j - i > 4) else (j - i) * (0.1 if k == 0 else 1.0)  # noqa: E731
    sol = solve(cost, 10, 2, allow_empty=False)
    assert sol.ranges[0][1] - sol.ranges[0][0] <= 4


def test_infeasible_returns_none():
    assert solve(lambda k, i, j: None if j - i > 2 else 1.0, 10, 3) is None
    assert solve(lambda k, i, j: 1.0, 3, 5, allow_empty=False) is None
    assert solve(lambda k, i, j: 1.0, 0, 2) is None


def test_more_workers_never_worse():
    rng = random.Random(3)
    unit = [rng.uniform(1, 5) for _ in range(10)]
    cost = lambda k, i, j: sum(unit[i:j])  # noqa: E731
    previous = float("inf")
    for m in range(1, 6):
        b = solve(cost, 10, m).bottleneck
        assert b <= previous + 1e-12
        previous = b


def test_check_ranges_messages():
    assert check_ranges([(0, 3), (3, 5)], 5) == []
    assert "gap or overlap" in check_ranges([(0, 3), (4, 5)], 5)[0]
    assert "gap or overlap" in check_ranges([(0, 3), (2, 5)], 5)[0]
    assert "expected 5" in check_ranges([(0, 3)], 5)[0]
