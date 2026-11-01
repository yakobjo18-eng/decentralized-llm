"""Bottleneck-minimizing contiguous chain partitioning.

Problem. Units 0..U-1 form a fixed sequential chain (the model's layer
order). Workers 0..m-1 are taken in a fixed pipeline order. Worker k
receives a contiguous range [s_k, e_k) with s_0 = 0, e_k = s_{k+1},
e_{m-1} = U. A range may be empty (the worker is unused) when
allow_empty is set. cost(k, i, j) is the estimated time for worker k to
run units [i, j), or None if those units don't fit in its memory.
Objective: minimize max_k cost(k, s_k, e_k) -- the pipeline's slowest
stage, which bounds its steady-state throughput -- not equal layer
counts.

Method: exact dynamic programming, two passes.
  1. B[k][j] = min over i <= j of max(B[k-1][i], cost(k-1, i, j)),
     B[0][0] = 0. The optimal bottleneck is B* = B[m][U].
  2. Among partitions whose every stage costs <= B*, minimize the SUM
     of stage costs (a tie-break that prefers not to waste work), with
     ties broken toward the earliest split point. Pass 2 is exact for
     that secondary objective because the constraint (stage <= B*) is
     per-stage, so optimal substructure holds -- unlike a single pass
     over (max, sum) pairs, which is not decomposable.

Complexity: O(m * U^2) cost lookups per pass (U = 15 units for GPT-2
small; m = workers). Memory O(m * U).

Why not shortest paths (e.g. Dijkstra): there is no routing choice --
the layer order and the worker order are fixed, and the objective is a
max over stages, not an additive path length. This is the classic
linear-partition / chain-to-chain mapping problem, which DP solves
exactly. Worker ORDER is searched separately (api.py) by trying every
permutation for small worker counts.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Callable, List, Optional, Sequence, Tuple

INF = math.inf
CostFn = Callable[[int, int, int], Optional[float]]


@dataclasses.dataclass(frozen=True)
class Solution:
    bottleneck: float
    total: float
    # One (start, end) per worker, in pipeline order; start == end means
    # the worker gets nothing.
    ranges: Tuple[Tuple[int, int], ...]
    stage_costs: Tuple[float, ...]


def _tolerant_le(a: float, b: float) -> bool:
    return a <= b + 1e-12 * max(1.0, abs(b))


def solve(cost: CostFn, n_units: int, n_workers: int, allow_empty: bool = True) -> Optional[Solution]:
    if n_units <= 0 or n_workers <= 0:
        return None
    if not allow_empty and n_workers > n_units:
        return None
    U, m = n_units, n_workers

    def c(k: int, i: int, j: int) -> Optional[float]:
        if i == j:
            return 0.0 if allow_empty else None
        return cost(k, i, j)

    # Pass 1: optimal bottleneck.
    best = [[INF] * (U + 1) for _ in range(m + 1)]
    best[0][0] = 0.0
    for k in range(1, m + 1):
        for j in range(U + 1):
            for i in range(j + 1):
                if best[k - 1][i] == INF:
                    continue
                ck = c(k - 1, i, j)
                if ck is None:
                    continue
                v = max(best[k - 1][i], ck)
                if v < best[k][j]:
                    best[k][j] = v
    b_star = best[m][U]
    if b_star == INF:
        return None

    # Pass 2: minimum total cost subject to every stage <= B*.
    tot = [[INF] * (U + 1) for _ in range(m + 1)]
    arg = [[-1] * (U + 1) for _ in range(m + 1)]
    tot[0][0] = 0.0
    for k in range(1, m + 1):
        for j in range(U + 1):
            for i in range(j + 1):
                if tot[k - 1][i] == INF:
                    continue
                ck = c(k - 1, i, j)
                if ck is None or not _tolerant_le(ck, b_star):
                    continue
                v = tot[k - 1][i] + ck
                if v < tot[k][j] - 1e-12 * max(1.0, abs(v)):
                    tot[k][j], arg[k][j] = v, i
    if tot[m][U] == INF:  # cannot happen if pass 1 succeeded; guard anyway
        return None

    ranges: List[Tuple[int, int]] = []
    j = U
    for k in range(m, 0, -1):
        i = arg[k][j]
        ranges.append((i, j))
        j = i
    ranges.reverse()
    costs = tuple(c(k, s, e) or 0.0 for k, (s, e) in enumerate(ranges))
    return Solution(bottleneck=max(costs), total=sum(costs), ranges=tuple(ranges), stage_costs=costs)


def brute_force(cost: CostFn, n_units: int, n_workers: int, allow_empty: bool = True) -> Optional[float]:
    """Exhaustive optimal bottleneck -- the test oracle. Exponential;
    only for tiny instances."""
    best = INF

    def rec(k: int, start: int, worst: float) -> None:
        nonlocal best
        if worst >= best:
            return
        if k == n_workers - 1:
            ck = (0.0 if allow_empty else None) if start == n_units else cost(k, start, n_units)
            if ck is not None:
                best = min(best, max(worst, ck))
            return
        for end in range(start if allow_empty else start + 1, n_units + 1):
            if start == end:
                ck = 0.0
            else:
                ck = cost(k, start, end)
                if ck is None:
                    continue
            rec(k + 1, end, max(worst, ck))

    rec(0, 0, 0.0)
    return None if best == INF else best


def check_ranges(ranges: Sequence[Tuple[int, int]], n_units: int) -> List[str]:
    """Structural check shared with the plan validator."""
    problems = []
    expected = 0
    for k, (s, e) in enumerate(ranges):
        if s != expected:
            problems.append(f"range {k} starts at {s}, expected {expected} (gap or overlap)")
        if e < s:
            problems.append(f"range {k} is reversed ({s}, {e})")
        expected = e
    if expected != n_units:
        problems.append(f"ranges end at {expected}, expected {n_units}")
    return problems
