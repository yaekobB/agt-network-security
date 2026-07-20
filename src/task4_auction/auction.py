# src/task4_auction/auction.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Hashable, List, Optional, Sequence, Tuple
import math
import random
import time

import networkx as nx

Node = Hashable


# ------------------------------------------------------------
# Data models (similar style to Task 3 planner.py)
# ------------------------------------------------------------
@dataclass(frozen=True)
class AuctionConfig:
    """
    Configuration for the secure-path auction.

    lam:
        Security penalty weight λ. Higher λ => planner avoids unsecure nodes more strongly.

    seed:
        Random seed used to generate agent prices (true costs).
    """
    lam: float = 5.0
    seed: int = 42
    price_min: int = 1
    price_max: int = 100
    exclude_endpoints: bool = True   # recommended: do not pay s/t


@dataclass
class AuctionResult:
    """
    Outputs for Task 4:
    - chosen path (allocation)
    - score breakdown
    - VCG payments + profits
    """
    s: Node
    t: Node
    path: List[Node]

    lam: float
    unsecure_count: int
    reported_cost_sum: float
    total_score: float
    runtime_ms: float

    secure_mask: Dict[Node, bool]
    true_costs: Dict[Node, int]
    reported_costs: Dict[Node, int]
    node_weights: Dict[Node, float]
    
    winners: List[Node]                 # paid nodes (usually path excluding endpoints)
    payments: Dict[Node, float]         # VCG payments (may contain inf if critical)
    profits: Dict[Node, float]          # payment - true_cost

    alt_cost_without: Dict[Node, float] # C_-i (best score when i removed); inf if disconnected
    alt_path_without: Dict[Node, Optional[List[Node]]]


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _build_secure_mask(G: nx.Graph, S: Sequence[Node]) -> Dict[Node, bool]:
    Sset = set(S)
    return {i: (i in Sset) for i in G.nodes()}


def _generate_true_costs(G: nx.Graph, cfg: AuctionConfig) -> Dict[Node, int]:
    rng = random.Random(cfg.seed)
    return {i: rng.randint(cfg.price_min, cfg.price_max) for i in G.nodes()}


def _apply_reports(true_costs: Dict[Node, int], liar_node: Optional[Node], liar_report: Optional[int]) -> Dict[Node, int]:
    """
    Reported costs = true costs, except optionally one liar node.
    This is ONLY for demonstration in UI; the mechanism uses reported costs.
    """
    rep = dict(true_costs)
    if liar_node is not None and liar_report is not None and liar_node in rep:
        rep[liar_node] = int(liar_report)
    return rep


def _node_weight(
    i: Node,
    s: Node,
    t: Node,
    secure_mask: Dict[Node, bool],
    reported_costs: Dict[Node, int],
    lam: float,
    exclude_endpoints: bool,
) -> float:
    """
    Weight of node i in the objective:
        w(i) = reported_cost(i) + lam * 1[i is unsecure]
    If exclude_endpoints=True, then w(s)=w(t)=0 (no payment/cost on endpoints).
    """
    if exclude_endpoints and (i == s or i == t):
        return 0.0
    unsecure = 0.0 if secure_mask.get(i, False) else 1.0
    return float(reported_costs[i]) + float(lam) * unsecure


def _directed_graph_with_enter_cost(
    G: nx.Graph,
    s: Node,
    t: Node,
    secure_mask: Dict[Node, bool],
    reported_costs: Dict[Node, int],
    lam: float,
    exclude_endpoints: bool,
) -> Tuple[nx.DiGraph, Dict[Node, float]]:
    """
    Convert an undirected graph G into a directed graph D where each directed edge (u->v)
    has weight = w(v). This makes the path cost equal to sum of node weights of visited nodes.
    """
    node_w = {
        i: _node_weight(i, s, t, secure_mask, reported_costs, lam, exclude_endpoints)
        for i in G.nodes()
    }

    D = nx.DiGraph()
    D.add_nodes_from(G.nodes())
    for u, v in G.edges():
        # entering v costs w(v); entering u costs w(u) in the reverse direction
        D.add_edge(u, v, weight=node_w[v])
        D.add_edge(v, u, weight=node_w[u])
    return D, node_w


def _shortest_path_score(
    G: nx.Graph,
    s: Node,
    t: Node,
    secure_mask: Dict[Node, bool],
    reported_costs: Dict[Node, int],
    lam: float,
    exclude_endpoints: bool,
) -> Tuple[List[Node], float, Dict[Node, float]]:
    """
    Compute the minimizing path P* and its total score C(P*).
    Returns (path, score, node_weights).
    """
    D, node_w = _directed_graph_with_enter_cost(G, s, t, secure_mask, reported_costs, lam, exclude_endpoints)
    # Dijkstra on directed graph
    score, path = nx.single_source_dijkstra(D, source=s, target=t, weight="weight")
    return path, float(score), node_w


def _count_unsecure_on_path(path: Sequence[Node], secure_mask: Dict[Node, bool], s: Node, t: Node, exclude_endpoints: bool) -> int:
    cnt = 0
    for i in path:
        if exclude_endpoints and (i == s or i == t):
            continue
        if not secure_mask.get(i, False):
            cnt += 1
    return cnt


# ------------------------------------------------------------
# VCG (Clarke pivot) payments
# ------------------------------------------------------------
def _vcg_payments(
    G: nx.Graph,
    s: Node,
    t: Node,
    path: Sequence[Node],
    base_cost: float,
    node_w: Dict[Node, float],
    secure_mask: Dict[Node, bool],
    reported_costs: Dict[Node, int],
    lam: float,
    exclude_endpoints: bool,
) -> Tuple[List[Node], Dict[Node, float], Dict[Node, float], Dict[Node, Optional[List[Node]]]]:
    """
    VCG payments for cost-minimization:
        pay_i = C_-i - (C* - w(i))

    where:
      - C* is the optimal score with all nodes
      - C_-i is the optimal score when node i is removed (not available)
      - w(i) is node i's weight on the chosen path

    If removing i disconnects s->t, then C_-i = inf and pay_i = inf (i is critical).
    Also store the alternative path (not just its cost).
    """
    # winners = path nodes excluding endpoints if configured
    winners = [
        i for i in path
        if not (exclude_endpoints and (i == s or i == t))
    ]

    payments: Dict[Node, float] = {}
    alt_cost_without: Dict[Node, float] = {}
    alt_path_without: Dict[Node, Optional[List[Node]]] = {}

    for i in winners:
        # Create a graph where node i is removed (agent unavailable)
        G2 = G.copy()
        if i in G2:
            G2.remove_node(i)

        try:
            path2, cost_without_i, _ = _shortest_path_score(
                G2, s, t, secure_mask, reported_costs, lam, exclude_endpoints
            )
            C_minus_i = float(cost_without_i)
            alt_path_without[i] = list(path2) if path2 else None
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            C_minus_i = float("inf")
            alt_path_without[i] = None

        alt_cost_without[i] = C_minus_i

        # Clarke pivot (externality) payment
        wi = float(node_w[i])
        if math.isinf(C_minus_i):
            payments[i] = float("inf")
        else:
            payments[i] = C_minus_i - (float(base_cost) - wi)

    return winners, payments, alt_cost_without, alt_path_without


# ------------------------------------------------------------
# One-shot runner used by UI
# ------------------------------------------------------------
def run_task4_auction(
    G: nx.Graph,
    S: Sequence[Node],
    s: Node,
    t: Node,
    cfg: AuctionConfig,
    liar_node: Optional[Node] = None,
    liar_report: Optional[int] = None,
) -> AuctionResult:
    """
    Task 4 entry point:
      1) build secure mask from S
      2) generate true costs
      3) optionally apply one liar report
      4) compute optimal secure path
      5) compute VCG payments and profits
    """
    t0 = time.perf_counter()

    secure_mask = _build_secure_mask(G, S)
    true_costs = _generate_true_costs(G, cfg)
    reported_costs = _apply_reports(true_costs, liar_node=liar_node, liar_report=liar_report)

    path, total_score, node_w = _shortest_path_score(
        G, s, t, secure_mask, reported_costs, cfg.lam, cfg.exclude_endpoints
    )

    unsecure_cnt = _count_unsecure_on_path(path, secure_mask, s, t, cfg.exclude_endpoints)

    # "monetary" sum = sum of reported costs on path (excluding endpoints if configured)
    monetary_sum = 0.0
    for i in path:
        if cfg.exclude_endpoints and (i == s or i == t):
            continue
        monetary_sum += float(reported_costs[i])

    winners, payments, alt_cost_wo, alt_path_without = _vcg_payments(
        G, s, t, path,
        base_cost=total_score,
        node_w=node_w,
        secure_mask=secure_mask,
        reported_costs=reported_costs,
        lam=cfg.lam,
        exclude_endpoints=cfg.exclude_endpoints,
    )

    profits: Dict[Node, float] = {}
    for i in winners:
        pay = payments[i]
        if math.isinf(pay):
            profits[i] = float("inf")
        else:
            profits[i] = float(pay) - float(true_costs[i])

    t1 = time.perf_counter()
    return AuctionResult(
        s=s,
        t=t,
        path=list(path),
        lam=float(cfg.lam),
        unsecure_count=int(unsecure_cnt),
        reported_cost_sum=float(monetary_sum),
        total_score=float(total_score),
        runtime_ms=(t1 - t0) * 1000.0,

        secure_mask=secure_mask,
        true_costs=true_costs,
        reported_costs=reported_costs,
        node_weights=node_w,

        winners=winners,
        payments=payments,
        profits=profits,
        alt_cost_without=alt_cost_wo,
        alt_path_without=alt_path_without,

    )
