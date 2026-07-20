"""
src/task2_coalitional/induce_set.py

Build a minimal Network Security Set induced by Shapley ranking.

Procedure:
1) Add nodes in descending Shapley order until set becomes a security set.
2) Prune to minimality by removing redundant nodes while preserving the security-set property.

We log both phases so UI can display steps and export CSV.

Optimization note:
- For UI/report explanation, intermediate minimality checks are useful.
- For CLI/batch experiments, intermediate minimality checks can be skipped by setting
  log_intermediate_minimality=False.
- Final security and final minimality are always checked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, List, Optional, Set

import networkx as nx

from src.security.checks import is_security_set, is_minimal_security_set, coverage_stats

Node = Hashable


@dataclass
class BuildStep:
    t: int
    added_node: Optional[Node]
    size_S: int
    is_security_set: bool
    is_minimal: bool


@dataclass
class PruneStep:
    step: int
    candidate_removed: Node
    removed: bool
    size_S_after: int
    still_security_set: bool
    is_minimal_after: bool


@dataclass
class InducedSetResult:
    S: Set[Node]
    converged: bool
    steps: List[BuildStep]
    pruned: bool
    final_is_security_set: bool
    final_is_minimal: bool
    size_S: int

    # extra metrics for report/UI
    satisfied_frac: float
    secured_edges_frac: float

    # prune/build transparency
    prune_steps: List[PruneStep]
    reached_security_at_size: Optional[int]
    S_ordered: List[Node]

    reached_security_at_step: Optional[int]
    S_build: Set[Node]
    S_build_ordered: List[Node]


def induce_minimal_security_set_from_ranking(
    G: nx.Graph,
    ranking: List[Node],
    prune: bool = True,
    log_intermediate_minimality: bool = True,
) -> InducedSetResult:
    """
    Build S by adding nodes in ranking order until security-set property holds.
    Then optionally prune S to minimality.

    Parameters:
        G:
            Input graph.
        ranking:
            Nodes ordered from highest to lowest Shapley value.
        prune:
            If True, remove redundant nodes after the build phase.
        log_intermediate_minimality:
            If True, compute minimality at intermediate build/prune log steps.
            This is useful for Streamlit explanations.

            If False, skip intermediate minimality checks for faster CLI/batch
            experiments. Final minimality is still always checked.

    Expected output:
    - S: final set
    - steps: build log
    - prune_steps: prune log
    - reached_security_at_size: where build first becomes security set
    - final checks: security set + minimal
    """
    build_steps: List[BuildStep] = []
    prune_steps: List[PruneStep] = []

    S: Set[Node] = set()
    t = 0

    reached_security_at_size: Optional[int] = None
    reached_security_at_step: Optional[int] = None

    S_build: Set[Node] = set()
    S_build_ordered: List[Node] = []
    current_build_order: List[Node] = []

    # -------------------------
    # Phase A: BUILD
    # -------------------------
    for node in ranking:
        t += 1
        S.add(node)
        current_build_order.append(node)

        sec = is_security_set(G, S)

        minimal = (
            is_minimal_security_set(G, S)
            if sec and log_intermediate_minimality
            else False
        )

        build_steps.append(
            BuildStep(
                t=t,
                added_node=node,
                size_S=len(S),
                is_security_set=sec,
                is_minimal=minimal,
            )
        )

        if sec:
            reached_security_at_size = len(S)
            reached_security_at_step = t
            S_build = set(S)
            S_build_ordered = list(current_build_order)
            break

    reached = is_security_set(G, S)

    # -------------------------
    # Phase B: PRUNE
    # -------------------------
    pruned_any = False

    if prune and reached:
        step = 0
        changed = True

        # Try removing low-importance nodes first.
        # ranking is high -> low, so reversed(ranking) is low -> high.
        #
        # Whenever a node is successfully removed, restart the scan because
        # the structure of S has changed.
        while changed:
            changed = False

            for node in reversed(ranking):
                if node not in S:
                    continue

                step += 1

                S_try = set(S)
                S_try.remove(node)

                still_sec = is_security_set(G, S_try)

                if still_sec:
                    S = S_try
                    pruned_any = True
                    changed = True

                if log_intermediate_minimality:
                    current_sec = is_security_set(G, S)
                    is_min_after = (
                        is_minimal_security_set(G, S)
                        if current_sec
                        else False
                    )
                else:
                    is_min_after = False

                prune_steps.append(
                    PruneStep(
                        step=step,
                        candidate_removed=node,
                        removed=bool(still_sec),
                        size_S_after=len(S),
                        still_security_set=bool(still_sec),
                        is_minimal_after=bool(is_min_after),
                    )
                )

                if changed:
                    break

    # -------------------------
    # Final checks + stats
    # -------------------------
    # These final checks are always performed, independently from
    # log_intermediate_minimality. This preserves the project requirement that
    # the final Shapley-induced set must be a valid minimal NSS.
    final_sec = is_security_set(G, S)
    final_min = is_minimal_security_set(G, S) if final_sec else False

    cov = coverage_stats(G, S)

    # Stable final ordering according to the original Shapley ranking.
    S_ordered = [node for node in ranking if node in S]

    return InducedSetResult(
        S=S,
        converged=final_sec,
        steps=build_steps,
        pruned=pruned_any,
        final_is_security_set=final_sec,
        final_is_minimal=final_min,
        size_S=len(S),
        satisfied_frac=cov.fraction_satisfied_nodes,
        secured_edges_frac=cov.fraction_secured_edges,
        prune_steps=prune_steps,
        reached_security_at_size=reached_security_at_size,
        S_ordered=S_ordered,
        reached_security_at_step=reached_security_at_step,
        S_build=S_build,
        S_build_ordered=S_build_ordered,
    )