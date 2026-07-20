"""
scripts/01_generate_graph.py

Generates a graph based on config.yaml and saves it under outputs/graphs/.

Run:
  python scripts/01_generate_graph.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH so "import src.*" works when running scripts directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import json
from pathlib import Path

import networkx as nx

from src.utils.io import load_yaml
from src.utils.seed import set_seed
from src.graphs.generators import GraphSpec, generate_graph


def graph_summary(G: nx.Graph) -> dict:
    """Compute a small summary for printing and reporting."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    avg_degree = (2 * m / n) if n > 0 else 0.0
    connected = nx.is_connected(G) if n > 0 else False
    return {
        "nodes": n,
        "edges": m,
        "avg_degree": avg_degree,
        "connected": connected,
    }


def save_graph_outputs(G: nx.Graph, out_dir: Path) -> tuple[Path, Path]:
    """
    Save graph in two formats:
    - GraphML (easy to inspect / import)
    - JSON (simple {nodes, edges} structure)

    Returns:
        (graphml_path, json_path)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    graphml_path = out_dir / "graph.graphml"
    json_path = out_dir / "graph.json"

    nx.write_graphml(G, graphml_path)

    data = {
        "nodes": list(G.nodes()),
        "edges": [[int(u), int(v)] for u, v in G.edges()],
    }
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    return graphml_path, json_path


def main() -> None:
    cfg = load_yaml("config.yaml")

    seed = int(cfg.get("random_seed", 42))
    set_seed(seed)

    graphs_cfg = cfg.get("graphs", {})
    gtype = graphs_cfg.get("type", "regular")

    if gtype == "regular":
        params = graphs_cfg.get("regular", {})
        spec = GraphSpec(type="regular", n=int(params["n"]), k=int(params["k"]))
    elif gtype == "erdos_renyi":
        params = graphs_cfg.get("erdos_renyi", {})
        spec = GraphSpec(type="erdos_renyi", n=int(params["n"]), p=float(params["p"]))
    elif gtype == "barabasi_albert":
        params = graphs_cfg.get("barabasi_albert", {})
        spec = GraphSpec(type="barabasi_albert", n=int(params["n"]), m=int(params["m"]))
    else:
        raise ValueError(f"Invalid graphs.type in config.yaml: {gtype}")

    G = generate_graph(spec)
    summ = graph_summary(G)

    print("=== Graph Generated ===")
    print(f"Graph type: {spec.type}")
    print(f"Nodes: {summ['nodes']}, Edges: {summ['edges']}")
    print(f"Avg degree: {summ['avg_degree']:.2f}")
    print(f"Connected: {summ['connected']}")

    out_dir = Path("outputs") / "graphs"
    graphml_path, json_path = save_graph_outputs(G, out_dir)

    print(f"Saved: {graphml_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
