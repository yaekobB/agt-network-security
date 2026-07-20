"""
src/task2_coalitional/__init__.py

Public API for Task 2 (Coalitional / Shapley).

Revised Task 2 uses non-negative characteristic functions:
- v1: node satisfaction
- v2: node + edge security
- v3: size-efficient security
"""

from .characteristic import (
    CharParams,
    CoalitionStats,
    coalition_stats,
    score_components,
    get_characteristic,
    v1_node_satisfaction,
    v2_node_edge_security,
    v3_size_efficient_security,
)

from .shapley import (
    ShapleyResult,
    approximate_shapley,
)

from .induce_set import (
    BuildStep,
    PruneStep,
    InducedSetResult,
    induce_minimal_security_set_from_ranking,
)

__all__ = [
    # characteristic
    "CharParams",
    "CoalitionStats",
    "coalition_stats",
    "score_components",
    "get_characteristic",
    "v1_node_satisfaction",
    "v2_node_edge_security",
    "v3_size_efficient_security",

    # shapley
    "ShapleyResult",
    "approximate_shapley",

    # induce set
    "BuildStep",
    "PruneStep",
    "InducedSetResult",
    "induce_minimal_security_set_from_ranking",
]