# src/task2_coalitional/compare_task1_task2.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import time

import networkx as nx

from src.security.checks import is_security_set, is_minimal_security_set, coverage_stats
from src.task2_coalitional.characteristic import CharParams, get_characteristic
from src.task2_coalitional.shapley import approximate_shapley
from src.task2_coalitional.induce_set import induce_minimal_security_set_from_ranking


@dataclass
class CompareRow:
    method: str
    size_S: int
    is_security_set: bool
    is_minimal_security_set: bool
    satisfied_frac: float
    secured_edges_frac: float
    runtime_ms: float
    extra: Dict[str, Any]


def run_task2_variant(
    G: nx.Graph,
    char_name: str,
    params2: CharParams,
    K: int,
    seed: int,
) -> CompareRow:
    """
    Run one Task 2 Shapley-induced NSS variant.

    This function:
    1. selects the characteristic function v(C),
    2. approximates Shapley values by Monte Carlo permutations,
    3. builds a security set from the Shapley ranking,
    4. prunes it to enforce inclusion-wise minimality,
    5. reports validation metrics.

    The revised characteristic functions are non-negative:
    - v1: node satisfaction
    - v2: node + edge security
    - v3: size-efficient security
    """
    v_fn = get_characteristic(char_name)

    t0 = time.perf_counter()

    shap = approximate_shapley(
        G,
        v_fn,
        params2,
        samples=int(K),
        seed=int(seed),
    )

    induced = induce_minimal_security_set_from_ranking(
        G,
        shap.ranking,
        prune=True,
    )

    t1 = time.perf_counter()

    S = induced.S
    cov = coverage_stats(G, S)

    return CompareRow(
        method=f"Shapley({char_name})",
        size_S=len(S),
        is_security_set=is_security_set(G, S),
        is_minimal_security_set=is_minimal_security_set(G, S),
        satisfied_frac=cov.fraction_satisfied_nodes,
        secured_edges_frac=cov.fraction_secured_edges,
        runtime_ms=1000.0 * (t1 - t0),
        extra={
            "K": int(K),
            "alpha": float(params2.alpha),
            "gamma": float(params2.gamma),
        },
    )