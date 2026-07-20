"""
src/graphs/generators.py

Graph generation utilities for:
- k-regular graphs
- Erdős–Rényi graphs
- Barabási–Albert graphs

All graphs returned are simple undirected graphs (networkx.Graph).
"""

from __future__ import annotations


from dataclasses import dataclass
from typing import Literal

import networkx as nx


GraphType = Literal["regular", "erdos_renyi", "barabasi_albert"]


@dataclass(frozen=True)
class GraphSpec:
    """Configuration object describing which graph to generate."""
    type: GraphType
    n: int
    # Params depending on type:
    k: int | None = None   # regular
    p: float | None = None # ER
    m: int | None = None   # BA
    seed: int | None = None



def generate_graph(spec: GraphSpec) -> nx.Graph:
    """
    Generate a graph according to GraphSpec.

    Returns:
        networkx.Graph

    Raises:
        ValueError: if parameters are invalid for the given type.
    """
    if spec.n <= 0:
        raise ValueError("n must be > 0")

    if spec.type == "regular":
        if spec.k is None:
            raise ValueError("regular graph requires k")
        if spec.k < 0 or spec.k >= spec.n:
            raise ValueError("regular graph requires 0 <= k < n")
        if (spec.n * spec.k) % 2 != 0:
            # k-regular graph exists only if n*k is even
            raise ValueError("regular graph requires n*k to be even")
        #G = nx.random_regular_graph(d=spec.k, n=spec.n)
        G = nx.random_regular_graph(d=spec.k, n=spec.n, seed=spec.seed)


    elif spec.type == "erdos_renyi":
        if spec.p is None:
            raise ValueError("erdos_renyi graph requires p")
        if not (0.0 <= spec.p <= 1.0):
            raise ValueError("ER graph requires 0 <= p <= 1")
        #G = nx.erdos_renyi_graph(n=spec.n, p=spec.p)
        G = nx.erdos_renyi_graph(n=spec.n, p=spec.p, seed=spec.seed)


    elif spec.type == "barabasi_albert":
        if spec.m is None:
            raise ValueError("barabasi_albert graph requires m")
        if spec.m <= 0 or spec.m >= spec.n:
            raise ValueError("BA graph requires 1 <= m < n")
        #G = nx.barabasi_albert_graph(n=spec.n, m=spec.m)
        G = nx.barabasi_albert_graph(n=spec.n, m=spec.m, seed=spec.seed)


    else:
        raise ValueError(f"Unknown graph type: {spec.type}")

    # Normalize to an integer node labeling 0..n-1
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")
    return G
