"""
src/security/checks.py

Core validators for the AGT project.

Project definition (Network Security Set):
A set S of nodes is a network security set if, for every node v:
    v is in S  OR  all neighbors of v are in S.

Minimality:
S is minimal if S is a security set and removing any node from S breaks the property.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Set, Hashable

import networkx as nx


Node = Hashable


def _to_set(nodes: Iterable[Node]) -> Set[Node]:
    """Convert any iterable of nodes to a set."""
    return set(nodes)


def is_security_set(G: nx.Graph, S: Iterable[Node]) -> bool:
    """
    Check whether S is a Network Security Set according to the professor's definition.

    Condition:
      For every node v in V:
        (v in S) OR (N(v) ⊆ S)

    Note:
      If a node has no neighbors (isolated node), then N(v) is empty, and empty ⊆ S is True.
      So isolated nodes are automatically "secured" even if not in S.
      (This is consistent with the logical definition.)
    """
    S_set = _to_set(S)

    for v in G.nodes():
        if v in S_set:
            continue
        # If v is not in S, then ALL neighbors must be in S
        for u in G.neighbors(v):
            if u not in S_set:
                return False
    return True


def is_minimal_security_set(G: nx.Graph, S: Iterable[Node]) -> bool:
    """
    Check whether S is a *minimal* network security set.

    Minimal means:
      - S is a security set
      - For every i in S: (S \\ {i}) is NOT a security set
    """
    S_set = _to_set(S)

    if not is_security_set(G, S_set):
        return False

    for i in list(S_set):
        S_minus = set(S_set)
        S_minus.remove(i)
        if is_security_set(G, S_minus):
            # If removing i still keeps it a security set, then S is not minimal
            return False
    return True


@dataclass(frozen=True)
class CoverageStats:
    """
    Useful descriptive statistics for a set S on a graph G.
    These are NOT additional constraints; they are metrics to report/compare solutions.
    """
    n_nodes: int
    n_edges: int
    size_S: int
    satisfied_nodes: int
    secured_edges: int
    fraction_satisfied_nodes: float
    fraction_secured_edges: float


def coverage_stats(G: nx.Graph, S: Iterable[Node]) -> CoverageStats:
    """
    Compute coverage metrics for set S.

    - A node v is "satisfied" if:
        v in S OR N(v) ⊆ S
      (exactly the security-set rule)

    - An edge (u, v) is "secured" if at least one endpoint is in S
      (this matches the intuition in the statement: traffic on edges controlled by a node in S)
    """
    S_set = _to_set(S)
    n = G.number_of_nodes()
    m = G.number_of_edges()

    # Count satisfied nodes
    satisfied = 0
    for v in G.nodes():
        if v in S_set:
            satisfied += 1
        else:
            # all neighbors in S?
            ok = True
            for u in G.neighbors(v):
                if u not in S_set:
                    ok = False
                    break
            if ok:
                satisfied += 1

    # Count secured edges: at least one endpoint in S
    secured_edges = 0
    for u, v in G.edges():
        if (u in S_set) or (v in S_set):
            secured_edges += 1

    frac_nodes = satisfied / n if n > 0 else 0.0
    frac_edges = secured_edges / m if m > 0 else 0.0

    return CoverageStats(
        n_nodes=n,
        n_edges=m,
        size_S=len(S_set),
        satisfied_nodes=satisfied,
        secured_edges=secured_edges,
        fraction_satisfied_nodes=frac_nodes,
        fraction_secured_edges=frac_edges,
    )
