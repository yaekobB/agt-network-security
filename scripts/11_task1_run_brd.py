"""
scripts/11_task1_run_brd.py

Runs Best Response Dynamics and prints:
- convergence status
- final |S|
- is_security_set
- is_minimal_security_set
- is_PNE
"""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.security.checks import is_security_set, is_minimal_security_set
from src.task1_strategic.game import default_params, set_from_actions, is_pure_nash_equilibrium
from src.task1_strategic.dynamics import init_actions, brd_run


def main() -> None:
    graph_path = Path("outputs") / "graphs" / "graph.graphml"
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph not found: {graph_path}. Run scripts/01_generate_graph.py first.")

    G = nx.read_graphml(graph_path)
    try:
        G = nx.relabel_nodes(G, {n: int(n) for n in G.nodes()})
    except ValueError:
        pass

    params = default_params(G, c=1.0, penalty=100.0)

    # Try a couple of initializations
    inits = [
        ("zeros", init_actions(G, "zeros")),
        ("random_p0.3", init_actions(G, "random", p=0.3, seed=42)),
    ]

    print("=== Task 1 - Step 1.2: Best Response Dynamics ===")
    print(f"Nodes={G.number_of_nodes()} Edges={G.number_of_edges()}")
    print("Params: cost=1.0, penalty=100.0\n")

    for name, a0 in inits:
        res = brd_run(G, a0, params, max_iters=500, shuffle_each_round=True, seed=42)
        S = set_from_actions(res.final_actions)

        sec = is_security_set(G, S)
        minimal = is_minimal_security_set(G, S)
        pne = is_pure_nash_equilibrium(G, res.final_actions, params)

        print(f"--- Init: {name} ---")
        print(f"Converged: {res.converged} in {res.iterations} iterations")
        print(f"|S|={len(S)}")
        print(f"is_security_set: {sec}")
        print(f"is_minimal_security_set: {minimal}")
        print(f"is_PNE: {pne}")
        print(f"Last changes count: {res.history_changed[-1] if res.history_changed else 'NA'}")
        print()

if __name__ == "__main__":
    main()
