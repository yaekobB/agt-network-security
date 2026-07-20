"""
src/task2_coalitional/shapley.py

Monte Carlo approximation of Shapley values for Task 2.

Shapley values measure each node's average marginal contribution:
phi_i = E_pi [ v(P_i ∪ {i}) - v(P_i) ]
where P_i is the set of nodes before i in a random permutation pi.

We approximate using K random permutations.
We also use caching on v(C) because many coalitions repeat across samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Hashable, Iterable, List, Optional, Set

import time
import random
import networkx as nx

from .characteristic import CharParams

Node = Hashable
ValueFn = Callable[[nx.Graph, Iterable[Node], CharParams], float]


@dataclass
class ShapleyResult:
    """Outputs of Shapley approximation."""
    phi: Dict[Node, float]
    ranking: List[Node]          # nodes sorted by phi desc
    samples: int
    seed: int
    runtime_ms: float
    # optional diagnostics
    cache_size: int
    cache_hits: int


def approximate_shapley(
    G: nx.Graph,
    value_fn: ValueFn,
    params: CharParams,
    samples: int = 1000,
    seed: int = 42,
    nodes: Optional[List[Node]] = None,
) -> ShapleyResult:
    """
    Approximate Shapley values via Monte Carlo permutations.

    Expected output:
    - phi: node -> Shapley value
    - ranking: nodes sorted descending by Shapley value
    - runtime and cache statistics

    Note:
        This is an approximation. Exact Shapley computation is exponential
        in the number of nodes, so Monte Carlo sampling is appropriate here.
    """
    if samples < 1:
        raise ValueError("samples must be >= 1.")

    t0 = time.perf_counter()
    rnd = random.Random(seed)

    V = nodes if nodes is not None else list(G.nodes())
    n = len(V)

    phi = {i: 0.0 for i in V}

    # Cache v(C) to speed up repeated coalition evaluations.
    cache: Dict[frozenset, float] = {}
    cache_hits = 0

    def v_of(Cset: Set[Node]) -> float:
        nonlocal cache_hits
        key = frozenset(Cset)

        if key in cache:
            cache_hits += 1
            return cache[key]

        val = value_fn(G, Cset, params)
        cache[key] = val
        return val

    # Monte Carlo: sample permutations and accumulate marginal contributions.
    for _ in range(samples):
        perm = V[:]
        rnd.shuffle(perm)

        C: Set[Node] = set()
        vC = v_of(C)

        for i in perm:
            C2 = set(C)
            C2.add(i)

            vC2 = v_of(C2)
            phi[i] += (vC2 - vC)

            C = C2
            vC = vC2

    # Average marginal contributions.
    for i in V:
        phi[i] /= float(samples)

    # Stable ranking:
    # - primary key: higher Shapley value first
    # - secondary key: node id when comparable
    #
    # Since project graphs are normalized to integer labels, this is stable
    # and useful for reproducible reports/CSVs.
    try:
        ranking = sorted(V, key=lambda x: (-phi[x], x))
    except TypeError:
        ranking = sorted(V, key=lambda x: phi[x], reverse=True)

    runtime_ms = (time.perf_counter() - t0) * 1000.0

    return ShapleyResult(
        phi=phi,
        ranking=ranking,
        samples=samples,
        seed=seed,
        runtime_ms=runtime_ms,
        cache_size=len(cache),
        cache_hits=cache_hits,
    )