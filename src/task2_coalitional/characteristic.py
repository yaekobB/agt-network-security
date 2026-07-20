"""
src/task2_coalitional/characteristic.py

Task 2: Coalitional Game Theory characteristic functions v(C).

Coalition C ⊆ V represents the set of secured nodes.

Revised design after feedback:
- Characteristic functions are non-negative security scores.
- v1 measures direct node satisfaction according to the NSS rule.
- v2 combines node satisfaction with secured-edge coverage.
- v3 adds size efficiency using a multiplicative factor, not subtraction.

This avoids negative coalition values and makes each v(C) easier to interpret.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Hashable, Iterable, Set, Dict

import networkx as nx

from src.security.checks import is_security_set, is_minimal_security_set

Node = Hashable


@dataclass(frozen=True)
class CoalitionStats:
    """Descriptive statistics about a coalition C on graph G."""
    n_nodes: int
    n_edges: int
    size_C: int

    # NSS condition metrics
    satisfied_nodes: int
    violations: int

    # Edge-coverage metric
    secured_edges: int

    # Normalized fractions
    frac_satisfied_nodes: float
    frac_violations: float
    frac_secured_edges: float
    frac_size: float

    # Logical flags
    is_security_set: bool
    is_minimal_security_set: bool


def _to_set(C: Iterable[Node]) -> Set[Node]:
    """Convert an iterable coalition representation to a set."""
    return set(C)


@dataclass(frozen=True)
class CharParams:
    """
    Parameters for revised non-negative characteristic functions.

    alpha:
        Weight on node satisfaction versus edge coverage in v2 and v3.
        alpha = 1 focuses only on node satisfaction.
        alpha = 0 focuses only on edge coverage.

    gamma:
        Strength of the size-efficiency factor in v3.
        Larger gamma discourages larger coalitions more strongly through the
        multiplicative size-efficiency factor, while keeping the value non-negative.
    """
    alpha: float = 0.7
    gamma: float = 1.0


def validate_char_params(params: CharParams) -> None:
    """
    Validate characteristic-function parameters.

    This does not change the model; it only prevents invalid parameter settings.
    """
    if not (0.0 <= float(params.alpha) <= 1.0):
        raise ValueError("CharParams.alpha must be in [0, 1].")

    if float(params.gamma) < 0.0:
        raise ValueError("CharParams.gamma must be non-negative.")


def coalition_stats(G: nx.Graph, C: Iterable[Node]) -> CoalitionStats:
    """
    Compute coalition statistics.

    A node v is satisfied if:
        v ∈ C OR N(v) ⊆ C

    An edge (u,v) is secured if:
        u ∈ C OR v ∈ C

    This function intentionally includes formal flags such as is_security_set
    and is_minimal_security_set. These are useful for UI/report/debugging.
    During final Task 2 construction, minimality is enforced by the build-prune
    procedure in induce_set.py.
    """
    Cset = _to_set(C)
    n = G.number_of_nodes()
    m = G.number_of_edges()

    satisfied = 0
    violations = 0

    for v in G.nodes():
        if v in Cset:
            satisfied += 1
        else:
            if all(u in Cset for u in G.neighbors(v)):
                satisfied += 1
            else:
                violations += 1

    secured_edges = sum(1 for u, v in G.edges() if u in Cset or v in Cset)

    frac_sat = satisfied / n if n > 0 else 0.0
    frac_viol = violations / n if n > 0 else 0.0
    frac_edges = secured_edges / m if m > 0 else 0.0
    frac_size = len(Cset) / n if n > 0 else 0.0

    sec = is_security_set(G, Cset)
    minimal = is_minimal_security_set(G, Cset) if sec else False

    return CoalitionStats(
        n_nodes=n,
        n_edges=m,
        size_C=len(Cset),
        satisfied_nodes=satisfied,
        violations=violations,
        secured_edges=secured_edges,
        frac_satisfied_nodes=frac_sat,
        frac_violations=frac_viol,
        frac_secured_edges=frac_edges,
        frac_size=frac_size,
        is_security_set=sec,
        is_minimal_security_set=minimal,
    )


def size_efficiency(st: CoalitionStats, params: CharParams) -> float:
    """
    Multiplicative size-efficiency factor.

    eff(C) = 1 / (1 + gamma * |C| / |V|)

    This discourages large coalitions without making v(C) negative.
    """
    validate_char_params(params)
    return 1.0 / (1.0 + float(params.gamma) * st.frac_size)


def score_components(G: nx.Graph, C: Iterable[Node], params: CharParams) -> Dict[str, float]:
    """
    Return a non-negative breakdown of the components used in v1/v2/v3.

    These components are useful for UI, report explanation, and debugging.
    """
    validate_char_params(params)

    st = coalition_stats(G, C)

    node_satisfaction = st.frac_satisfied_nodes
    edge_coverage = st.frac_secured_edges
    size_eff = size_efficiency(st, params)

    combined_security = (
        params.alpha * node_satisfaction
        + (1.0 - params.alpha) * edge_coverage
    )

    return {
        "node_satisfaction": node_satisfaction,
        "edge_coverage": edge_coverage,
        "combined_security": combined_security,
        "violations_frac": st.frac_violations,
        "size_frac": st.frac_size,
        "size_efficiency": size_eff,
        "is_security_set": 1.0 if st.is_security_set else 0.0,
        "is_minimal_security_set": 1.0 if st.is_minimal_security_set else 0.0,
    }


def v1_node_satisfaction(G: nx.Graph, C: Iterable[Node], params: CharParams) -> float:
    """
    v1(C) = f_sat(C)

    Meaning:
        Fraction of nodes whose NSS condition is satisfied.

    Interpretation:
        v1(C) = 1 iff C is a network security set.
        v1(C) is always in [0,1].
    """
    validate_char_params(params)

    st = coalition_stats(G, C)
    return st.frac_satisfied_nodes


def v2_node_edge_security(G: nx.Graph, C: Iterable[Node], params: CharParams) -> float:
    """
    v2(C) = alpha * f_sat(C) + (1 - alpha) * f_edge(C)

    Meaning:
        Combines formal NSS node satisfaction with practical secured-edge coverage.

    Interpretation:
        High value means the coalition satisfies many nodes and secures many edges.
        v2(C) is always in [0,1].
    """
    validate_char_params(params)

    st = coalition_stats(G, C)
    return (
        params.alpha * st.frac_satisfied_nodes
        + (1.0 - params.alpha) * st.frac_secured_edges
    )


def v3_size_efficient_security(G: nx.Graph, C: Iterable[Node], params: CharParams) -> float:
    """
    v3(C) = v2(C) * 1 / (1 + gamma * |C| / |V|)

    Meaning:
        Rewards coalitions that provide high security with fewer selected nodes.

    Interpretation:
        This replaces the old subtractive size penalty.
        The value remains non-negative because the size factor is multiplicative.
    """
    validate_char_params(params)

    st = coalition_stats(G, C)

    combined_security = (
        params.alpha * st.frac_satisfied_nodes
        + (1.0 - params.alpha) * st.frac_secured_edges
    )

    return combined_security * size_efficiency(st, params)


def get_characteristic(name: str) -> Callable[[nx.Graph, Iterable[Node], CharParams], float]:
    """
    Return a characteristic function by name.

    Stable names:
        v1 -> node satisfaction
        v2 -> node + edge security
        v3 -> size-efficient security
    """
    name = name.strip().lower()

    if name in (
        "v1",
        "node_satisfaction",
        "v1_node_satisfaction",
    ):
        return v1_node_satisfaction

    if name in (
        "v2",
        "node_edge",
        "node_edge_security",
        "v2_node_edge_security",
    ):
        return v2_node_edge_security

    if name in (
        "v3",
        "size_efficient",
        "size_efficient_security",
        "v3_size_efficient_security",
    ):
        return v3_size_efficient_security

    raise ValueError(f"Unknown characteristic function: {name}")