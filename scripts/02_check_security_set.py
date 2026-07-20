"""
scripts/02_check_security_set.py

Loads the generated graph and tests the security-set definitions.

Run:
  python scripts/02_check_security_set.py
"""

from __future__ import annotations

import sys
from pathlib import Path
import random

import networkx as nx

# Ensure project root is on PYTHONPATH so "import src.*" works when running scripts directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.security.checks import is_security_set, is_minimal_security_set, coverage_stats


def pretty_stats(stats) -> str:
    return (
        f"size(S)={stats.size_S} | "
        f"satisfied_nodes={stats.satisfied_nodes}/{stats.n_nodes} "
        f"({stats.fraction_satisfied_nodes:.2%}) | "
        f"secured_edges={stats.secured_edges}/{stats.n_edges} "
        f"({stats.fraction_secured_edges:.2%})"
    )


def main() -> None:
    graph_path = Path("outputs") / "graphs" / "graph.graphml"
    if not graph_path.exists():
        raise FileNotFoundError(
            f"Graph file not found: {graph_path}. Run scripts/01_generate_graph.py first."
        )

    G = nx.read_graphml(graph_path)

    # GraphML reads node IDs as strings sometimes; convert back to ints when possible.
    try:
        G = nx.relabel_nodes(G, {n: int(n) for n in G.nodes()})
    except ValueError:
        # If conversion fails, keep as-is (still works)
        pass

    n = G.number_of_nodes()
    m = G.number_of_edges()

    print("=== Step 0.3: Security Set Checks ===")
    print(f"Loaded graph: {graph_path}")
    print(f"Nodes: {n}, Edges: {m}")

    # Test sets
    S_empty = set()
    S_all = set(G.nodes())

    # Random set (about 30% nodes)
    random.seed(42)
    S_rand = {v for v in G.nodes() if random.random() < 0.30}

    tests = [
        ("Empty set", S_empty),
        ("All nodes", S_all),
        ("Random 30%", S_rand),
    ]

    for name, S in tests:
        ok = is_security_set(G, S)
        minimal = is_minimal_security_set(G, S)
        stats = coverage_stats(G, S)

        print("\n---", name, "---")
        print(f"is_security_set: {ok}")
        print(f"is_minimal_security_set: {minimal}")
        print(pretty_stats(stats))


if __name__ == "__main__":
    main()
