# src/task3_planner/planner.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Hashable, List, Sequence, Tuple
import random
import time

import networkx as nx

Node = Hashable


# -----------------------------
# Data models
# -----------------------------
@dataclass(frozen=True)
class Player:
    node: Node
    budget: int  # 1..100


@dataclass(frozen=True)
class Vendor:
    vid: str
    price: int      # 1..100
    level: int      # 1..10
    capacity: int   # used for limited-items case


@dataclass(frozen=True)
class PlannerConfig:
    """
    Configuration for Task 3 vendor assignment.

    alpha:
        Weight of vendor security level in the normalized utility.
        alpha close to 1 means security level is more important.
        alpha close to 0 means affordability is more important.
    """
    n_vendors: int = 8
    alpha: float = 0.7
    cap_min: int = 1
    cap_max: int = 3
    seed: int = 42

    # Optional vendor generator:
    # if False, use fully random vendors as required by the project;
    # if True, generate vendors with a price-level trade-off.
    balanced_vendors: bool = False
    price_noise: float = 8.0


@dataclass
class PlannerResult:
    case_name: str
    assignments: Dict[Node, str]         # player -> vendor_id or "UNASSIGNED"
    utilities: Dict[Node, float]         # player -> u(i, assigned)
    total_welfare: float
    runtime_ms: float
    unmatched: List[Node]                # players assigned to UNASSIGNED
    vendor_loads: Dict[str, int]         # vendor_id -> number of assigned players


# -----------------------------
# Generation
# -----------------------------
def generate_players(S: Sequence[Node], seed: int) -> List[Player]:
    """
    Generate one player for each selected security node.

    Each player receives a random budget in [1, 100], as required by the project.
    """
    rng = random.Random(seed)
    return [Player(node=i, budget=rng.randint(1, 100)) for i in S]


def generate_vendors(M: int, seed: int, cap_min: int, cap_max: int) -> List[Vendor]:
    """
    Generate M vendors.

    Each vendor has:
    - price in [1, 100]
    - security level in [1, 10]
    - capacity in [cap_min, cap_max]
    """
    rng = random.Random(seed + 100_000)
    vendors: List[Vendor] = []

    for j in range(M):
        vendors.append(
            Vendor(
                vid=f"V{j}",
                price=rng.randint(1, 100),
                level=rng.randint(1, 10),
                capacity=rng.randint(cap_min, cap_max),
            )
        )

    return vendors


def generate_vendors_balanced(
    M: int,
    seed: int,
    cap_min: int,
    cap_max: int,
    price_noise: float = 8.0,
) -> List[Vendor]:
    """
    Optional vendor generator with a price/security trade-off.

    Higher security levels tend to have higher prices, with some noise.
    This is optional and should be presented as an experimental variant,
    not as the default project requirement.
    """
    rng = random.Random(seed + 100_000)
    vendors: List[Vendor] = []

    level_min, level_max = 1, 10
    price_min, price_max = 10, 100

    for j in range(M):
        level = rng.randint(level_min, level_max)

        t = (level - level_min) / float(level_max - level_min)
        base_price = price_min + t * (price_max - price_min)

        price = int(round(base_price + rng.gauss(0.0, float(price_noise))))
        price = max(1, min(100, price))

        vendors.append(
            Vendor(
                vid=f"V{j}",
                price=price,
                level=level,
                capacity=rng.randint(cap_min, cap_max),
            )
        )

    return vendors


# -----------------------------
# Utility + compatibility
# -----------------------------
def compatible(p: Player, v: Vendor) -> bool:
    """
    A vendor is compatible with a player if the vendor price does not exceed the player's budget.
    """
    return v.price <= p.budget


def utility(p: Player, v: Vendor, alpha: float) -> float:
    """
    Normalized non-negative utility for a compatible player-vendor pair.

    Precondition:
        v.price <= p.budget

    Utility:
        u(i,v) = alpha * (level_v / 10)
               + (1 - alpha) * (1 - price_v / budget_i)

    where:
        level_v / 10 is normalized security level;
        1 - price_v / budget_i is normalized affordability.

    For compatible pairs, utility is non-negative.
    If a pair is incompatible, it should not be evaluated.
    """
    if not compatible(p, v):
        raise ValueError("utility() called on an incompatible player-vendor pair")

    alpha = max(0.0, min(1.0, float(alpha)))

    security_score = float(v.level) / 10.0
    affordability_score = 1.0 - (float(v.price) / float(p.budget))

    return alpha * security_score + (1.0 - alpha) * affordability_score


# -----------------------------
# Case A: infinite items
# -----------------------------
def solve_infinite_items(
    players: Sequence[Player],
    vendors: Sequence[Vendor],
    alpha: float,
) -> PlannerResult:
    """
    Solve the infinite-capacity vendor assignment case.

    Since vendor supply is unlimited, each player can independently choose
    the compatible vendor with maximum utility. This maximizes total welfare
    because one player's choice does not affect another player's feasible choices.
    """
    t0 = time.perf_counter()

    assignments: Dict[Node, str] = {}
    utilities: Dict[Node, float] = {}
    vendor_loads: Dict[str, int] = {v.vid: 0 for v in vendors}

    for p in players:
        best_vid = "UNASSIGNED"
        best_u = 0.0

        for v in vendors:
            if not compatible(p, v):
                continue

            u = utility(p, v, alpha)

            if best_vid == "UNASSIGNED" or u > best_u:
                best_u = u
                best_vid = v.vid

        assignments[p.node] = best_vid
        utilities[p.node] = best_u if best_vid != "UNASSIGNED" else 0.0

        if best_vid != "UNASSIGNED":
            vendor_loads[best_vid] += 1

    unmatched = [i for i, vid in assignments.items() if vid == "UNASSIGNED"]
    welfare = sum(utilities.values())

    t1 = time.perf_counter()

    return PlannerResult(
        case_name="infinite_items",
        assignments=assignments,
        utilities=utilities,
        total_welfare=float(welfare),
        runtime_ms=(t1 - t0) * 1000.0,
        unmatched=unmatched,
        vendor_loads=vendor_loads,
    )


# -----------------------------
# Case B: limited items via min-cost flow
# -----------------------------
def solve_limited_items(
    players: Sequence[Player],
    vendors: Sequence[Vendor],
    alpha: float,
) -> PlannerResult:
    """
    Solve the limited-capacity vendor assignment case.

    Here player choices are coupled by vendor capacities, so independent greedy
    assignment is not sufficient. We formulate the problem as a min-cost flow:

    - each player sends one unit of flow;
    - compatible player-vendor edges have cost = -utility;
    - vendor-sink edges enforce capacities;
    - a dummy UNASSIGNED vendor keeps the problem feasible with utility 0.

    Minimizing total cost is equivalent to maximizing total welfare.
    """
    t0 = time.perf_counter()

    G = nx.DiGraph()
    src = "SRC"
    sink = "SINK"

    n_players = len(players)

    G.add_node(src, demand=-n_players)
    G.add_node(sink, demand=n_players)

    dummy_vid = "UNASSIGNED"
    dummy_capacity = n_players

    # Scale utilities to integer costs for NetworkX min_cost_flow.
    SCALE = 10_000

    # Source -> player edges
    for p in players:
        pn = f"P:{p.node}"
        G.add_node(pn, demand=0)
        G.add_edge(src, pn, capacity=1, weight=0)

    # Vendor -> sink edges
    for v in vendors:
        vn = f"V:{v.vid}"
        G.add_node(vn, demand=0)
        G.add_edge(vn, sink, capacity=int(v.capacity), weight=0)

    # Dummy unassigned vendor -> sink
    dummy_node = f"V:{dummy_vid}"
    G.add_node(dummy_node, demand=0)
    G.add_edge(dummy_node, sink, capacity=int(dummy_capacity), weight=0)

    # Player -> vendor edges
    for p in players:
        pn = f"P:{p.node}"

        for v in vendors:
            if not compatible(p, v):
                continue

            u = utility(p, v, alpha)
            cost = int(round(-u * SCALE))
            G.add_edge(pn, f"V:{v.vid}", capacity=1, weight=cost)

        # Always allow the player to remain unmatched with utility 0.
        G.add_edge(pn, dummy_node, capacity=1, weight=0)

    flow = nx.min_cost_flow(G)

    assignments: Dict[Node, str] = {}
    utilities: Dict[Node, float] = {}
    vendor_loads: Dict[str, int] = {v.vid: 0 for v in vendors}
    vendor_loads[dummy_vid] = 0

    vendor_map = {v.vid: v for v in vendors}

    for p in players:
        pn = f"P:{p.node}"
        chosen_vid = dummy_vid

        for to_node, flow_value in flow[pn].items():
            if flow_value != 1:
                continue
            chosen_vid = to_node.split(":", 1)[1]
            break

        assignments[p.node] = chosen_vid
        vendor_loads[chosen_vid] = vendor_loads.get(chosen_vid, 0) + 1

        if chosen_vid == dummy_vid:
            utilities[p.node] = 0.0
        else:
            utilities[p.node] = utility(p, vendor_map[chosen_vid], alpha)

    unmatched = [i for i, vid in assignments.items() if vid == dummy_vid]
    welfare = sum(utilities.values())

    t1 = time.perf_counter()

    return PlannerResult(
        case_name="limited_items",
        assignments=assignments,
        utilities=utilities,
        total_welfare=float(welfare),
        runtime_ms=(t1 - t0) * 1000.0,
        unmatched=unmatched,
        vendor_loads=vendor_loads,
    )


# -----------------------------
# One-shot runner used by UI
# -----------------------------
def run_task3_planner(
    S: Sequence[Node],
    cfg: PlannerConfig,
) -> Tuple[List[Player], List[Vendor], PlannerResult, PlannerResult]:
    """
    Run Task 3 from a selected security set S.

    Steps:
    1. generate players from S;
    2. generate vendors;
    3. solve infinite-capacity assignment;
    4. solve limited-capacity assignment.
    """
    S_list = list(S)

    players = generate_players(S_list, seed=cfg.seed)

    if getattr(cfg, "balanced_vendors", False):
        vendors = generate_vendors_balanced(
            cfg.n_vendors,
            seed=cfg.seed,
            cap_min=cfg.cap_min,
            cap_max=cfg.cap_max,
            price_noise=getattr(cfg, "price_noise", 8.0),
        )
    else:
        vendors = generate_vendors(
            cfg.n_vendors,
            seed=cfg.seed,
            cap_min=cfg.cap_min,
            cap_max=cfg.cap_max,
        )

    res_A = solve_infinite_items(players, vendors, alpha=cfg.alpha)
    res_B = solve_limited_items(players, vendors, alpha=cfg.alpha)

    return players, vendors, res_A, res_B