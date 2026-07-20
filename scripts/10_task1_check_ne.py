"""
scripts/10_task1_check_ne.py

Task 1 - Step 1.1 smoke test:
- load graph
- test action profiles
- check security-set validity + minimality
- check Pure Nash Equilibrium (PNE) under our defined strategic game
"""

from __future__ import annotations

import sys
from pathlib import Path
import random

import networkx as nx

# Ensure project root import works
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.security.checks import is_security_set, is_minimal_security_set
from src.task1_strategic.game import (
    actions_from_set,
    default_params,
    is_pure_nash_equilibrium,
    set_from_actions,
)

def main() -> None:
    graph_path = Path("outputs") / "graphs" / "graph.graphml"
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph not found: {graph_path}. Run scripts/01_generate_graph.py first.")

    G = nx.read_graphml(graph_path)

    # convert node labels to int if possible
    try:
        G = nx.relabel_nodes(G, {n: int(n) for n in G.nodes()})
    except ValueError:
        pass

    params = default_params(G, c=1.0, penalty=100.0)

    # Define some test sets S
    S_empty = set()
    S_all = set(G.nodes())
    random.seed(42)
    S_rand = {v for v in G.nodes() if random.random() < 0.30}

    tests = [
        ("Empty", S_empty),
        ("All", S_all),
        ("Random30%", S_rand),
    ]

    print("=== Task 1 - Step 1.1: Game & PNE Checks ===")
    print(f"Nodes={G.number_of_nodes()} Edges={G.number_of_edges()}")
    print(f"Params: cost=1.0, penalty=100.0\n")

    for name, S in tests:
        a = actions_from_set(G, S)

        sec = is_security_set(G, S)
        minimal = is_minimal_security_set(G, S)
        pne = is_pure_nash_equilibrium(G, a, params)

        print(f"--- {name} ---")
        print(f"|S|={len(S)}")
        print(f"is_security_set: {sec}")
        print(f"is_minimal_security_set: {minimal}")
        print(f"is_PNE: {pne}")

        # sanity: action->set conversion
        S2 = set_from_actions(a)
        if S2 != set(S):
            print("WARNING: action->set mismatch (labeling issue).")

        print()

if __name__ == "__main__":
    main()
