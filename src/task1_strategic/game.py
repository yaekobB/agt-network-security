"""
src/task1_strategic/game.py

Task 1 (Strategic / Non-cooperative):
- Players are nodes of the graph.
- Each player chooses action a_i in {0,1}:
    1 = Secure (join S)
    0 = Not secure (not in S)

Safety condition (mirrors project rule):
- If a_i = 0, player i is "safe" only if ALL neighbors of i are secure (a_j = 1 for all j in N(i)).
- If a_i = 0 and there exists a neighbor with 0, then i is "unsafe".

Utility design:
- If secure: u_i = -c_i
- If not secure and safe: u_i = 0
- If not secure and unsafe: u_i = -P

With P >> c_i, players prefer to secure when unsafe, but prefer free-riding (a_i=0) when safe.
This tends to produce security sets, and discourages "everyone secures" minimality violations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Hashable, Iterable, List, Mapping, Tuple

import networkx as nx

Node = Hashable
ActionProfile = Dict[Node, int]  # node -> 0/1


@dataclass(frozen=True)
class GameParams:
    """Parameters for the strategic game."""
    cost: Mapping[Node, float]   # c_i > 0
    penalty: float               # P (large)


def actions_from_set(G: nx.Graph, S: Iterable[Node]) -> ActionProfile:
    """Convert a set S to an action profile: a_i=1 iff i in S."""
    Sset = set(S)
    return {v: (1 if v in Sset else 0) for v in G.nodes()}


def set_from_actions(a: ActionProfile) -> set[Node]:
    """Convert action profile to set S of secured nodes."""
    return {i for i, ai in a.items() if ai == 1}


def is_safe_if_not_secure(G: nx.Graph, i: Node, a: ActionProfile) -> bool:
    """
    Safety condition if i plays 0:
    safe iff all neighbors of i play 1.
    """
    for j in G.neighbors(i):
        if a.get(j, 0) != 1:
            return False
    return True


def payoff_i(G: nx.Graph, i: Node, a: ActionProfile, params: GameParams) -> float:
    """
    Compute utility of player i under action profile a.
    """
    ai = a[i]
    if ai == 1:
        return -float(params.cost[i])  # pay cost
    # ai == 0
    if is_safe_if_not_secure(G, i, a):
        return 0.0
    return -float(params.penalty)


def best_response(G: nx.Graph, i: Node, a: ActionProfile, params: GameParams) -> int:
    """
    Best response of i against others' actions a_{-i}.
    Returns 0 or 1.

    Tie-breaking: if equal utility, prefer 0 (free-ride).
    """
    a0 = dict(a); a0[i] = 0
    a1 = dict(a); a1[i] = 1

    u0 = payoff_i(G, i, a0, params)
    u1 = payoff_i(G, i, a1, params)

    if u1 > u0:
        return 1
    return 0  # tie or u0 better


def unilateral_deviation_profitable(
    G: nx.Graph, i: Node, a: ActionProfile, params: GameParams
) -> Tuple[bool, float, float]:
    """
    Check if player i can improve by deviating (0<->1).
    Returns (profitable?, current_u, deviated_u_best).
    """
    current_u = payoff_i(G, i, a, params)
    alt = 1 - a[i]
    a_alt = dict(a); a_alt[i] = alt
    alt_u = payoff_i(G, i, a_alt, params)
    return (alt_u > current_u, current_u, alt_u)


def is_pure_nash_equilibrium(G: nx.Graph, recognizer_a: ActionProfile, params: GameParams) -> bool:
    """
    A profile a is a Pure Nash Equilibrium if no player can profitably deviate unilaterally.
    """
    a = dict(recognizer_a)
    for i in G.nodes():
        profitable, _, _ = unilateral_deviation_profitable(G, i, a, params)
        if profitable:
            return False
    return True


def payoffs(G: nx.Graph, a: ActionProfile, params: GameParams) -> Dict[Node, float]:
    """Return payoffs for all players."""
    return {i: payoff_i(G, i, a, params) for i in G.nodes()}

def player_utility(G: nx.Graph, i: Node, a: ActionProfile, params: GameParams) -> float:
    """
    Wrapper used by learning dynamics (e.g., regret matching).
    Utility == payoff of player i.
    """
    return payoff_i(G, i, a, params)


def default_params(G: nx.Graph, c: float = 1.0, penalty: float = 100.0) -> GameParams:
    """
    Convenient defaults:
    - uniform cost c_i = c
    - penalty P = penalty (should be >> c)
    """
    costs = {i: float(c) for i in G.nodes()}
    return GameParams(cost=costs, penalty=float(penalty))
