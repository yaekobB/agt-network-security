from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Project imports
from src.graphs.generators import GraphSpec, generate_graph
from src.task1_strategic.game import (
    default_params,
    set_from_actions,
    is_pure_nash_equilibrium,
)
from src.task1_strategic.dynamics import init_actions, brd_run
from src.task1_strategic.regret_matching import regret_matching_run
from src.task1_strategic.fictitious_play import fictitious_play_run
from src.security.checks import is_security_set, is_minimal_security_set, coverage_stats


def make_graph(args) -> tuple[nx.Graph, str]:
    """Generate graph and return graph plus readable parameter string."""

    if args.graph == "regular":
        spec = GraphSpec(type="regular", n=args.n, k=args.k, seed=args.seed)
        gparam = f"regular(n={args.n},k={args.k})"

    elif args.graph == "er":
        if args.p is None:
            p = 4.0 / max(args.n - 1, 1)
        else:
            p = float(args.p)

        spec = GraphSpec(type="erdos_renyi", n=args.n, p=p, seed=args.seed)
        gparam = f"erdos_renyi(n={args.n},p={p:.6f})"

    elif args.graph == "ba":
        spec = GraphSpec(type="barabasi_albert", n=args.n, m=args.m, seed=args.seed)
        gparam = f"barabasi_albert(n={args.n},m={args.m})"

    else:
        raise ValueError(f"Unknown graph type: {args.graph}")

    G = generate_graph(spec)

    # Keep node labels integer when possible, for cleaner CSVs and consistency
    # with UI/report examples.
    try:
        G = nx.relabel_nodes(G, {v: int(v) for v in G.nodes()})
    except Exception:
        pass

    return G, gparam


def summarize_result(
    *,
    method: str,
    G: nx.Graph,
    params,
    result,
    runtime_ms: float,
    graph_label: str,
    init_mode: str,
    init_p: float | None,
    seed: int,
    max_iter: int,
    fp_mode: str | None = None,
    store_history: bool | None = None,
    check_every: int | None = None,
) -> Dict[str, Any]:
    """
    Build one clean summary row.

    This function intentionally computes only final-run statistics.
    It does not build full per-iteration logs, because the CLI is intended
    for summary-only experiments, especially on larger graphs.
    """

    S = set_from_actions(result.final_actions)
    stats = coverage_stats(G, S)

    pne = is_pure_nash_equilibrium(G, result.final_actions, params)
    sec = is_security_set(G, S)
    minimal = is_minimal_security_set(G, S)

    return {
        "seed": seed,
        "graph": graph_label,
        "n": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "init_mode": init_mode,
        "init_p": init_p,
        "method": method,
        "fp_mode": fp_mode,
        "max_iter": max_iter,
        # CLI optimization metadata.
        # store_history=False means RM/FP did not store every full action profile.
        # check_every controls how often RM/FP checked the global PNE condition.
        "store_history": store_history,
        "check_every": check_every,
        "reached_pne": bool(pne),
        "iterations": int(result.iterations),
        "final_size_S": len(S),
        "size_ratio": len(S) / max(G.number_of_nodes(), 1),
        "is_security_set": bool(sec),
        "is_minimal_security_set": bool(minimal),
        "satisfied_frac": stats.fraction_satisfied_nodes,
        "secured_edges_frac": stats.fraction_secured_edges,
        "runtime_ms": runtime_ms,
        "runtime_sec": runtime_ms / 1000.0,
    }


def run_one_method(args, G: nx.Graph, graph_label: str, method: str) -> Dict[str, Any]:
    """Run one Task 1 method and return one summary row."""

    params = default_params(G, c=1.0, penalty=100.0)

    if args.init == "random":
        a0 = init_actions(G, "random", p=args.init_p, seed=args.seed)
        init_p_used = args.init_p
    else:
        a0 = init_actions(G, args.init, seed=args.seed)
        init_p_used = None

    method = method.lower()
    t0 = time.perf_counter()

    if method == "brd":
        # BRD usually converges quickly and is mainly used as the practical
        # baseline optimizer. We keep its existing history behavior unchanged.
        result = brd_run(
            G,
            a0,
            params,
            max_iters=args.brd_iters,
            seed=args.seed,
            shuffle_each_round=True,
        )
        max_iter = args.brd_iters
        method_name = "BRD"
        fp_mode = None

    elif method == "rm":
        result = regret_matching_run(
            G,
            a0,
            params,
            T=args.rm_iters,
            seed=args.seed,
            sequential=args.rm_sequential,
            shuffle_each_round=True,
            # CLI optimization:
            # Do not store full per-iteration action profiles unless explicitly
            # requested. This saves memory on large graphs.
            store_history=args.store_history,
            # PNE validation frequency:
            # check_every=1 gives exact checked stopping; larger values reduce
            # repeated global PNE-check overhead.
            check_every=args.check_every,
        )
        max_iter = args.rm_iters
        method_name = "RM-seq" if args.rm_sequential else "RM"
        fp_mode = None

    elif method == "fp":
        result = fictitious_play_run(
            G,
            a0,
            params,
            T=args.fp_iters,
            seed=args.seed,
            sequential=args.fp_sequential,
            shuffle_each_round=True,
            # CLI optimization:
            # Do not store full per-iteration action profiles unless explicitly
            # requested. Streamlit keeps full histories, but CLI is summary-only.
            store_history=args.store_history,
            # PNE validation frequency:
            # check_every=1 gives exact checked stopping; larger values reduce
            # repeated global PNE-check overhead.
            check_every=args.check_every,
        )
        max_iter = args.fp_iters
        method_name = "FP-seq" if args.fp_sequential else "FP-sync"
        fp_mode = "sequential" if args.fp_sequential else "synchronous"

    else:
        raise ValueError(f"Unknown method: {method}")

    runtime_ms = (time.perf_counter() - t0) * 1000.0

    return summarize_result(
        method=method_name,
        G=G,
        params=params,
        result=result,
        runtime_ms=runtime_ms,
        graph_label=graph_label,
        init_mode=args.init,
        init_p=init_p_used,
        seed=args.seed,
        max_iter=max_iter,
        fp_mode=fp_mode,
        store_history=args.store_history,
        check_every=args.check_every,
    )


def safe_name(text: str) -> str:
    return (
        str(text)
        .replace(" ", "")
        .replace("(", "_")
        .replace(")", "")
        .replace(",", "_")
        .replace("=", "")
        .replace(".", "p")
        .replace("/", "_")
    )


def save_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError("No rows to save.")

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run Task 1 experiments outside Streamlit. "
            "This CLI is designed for summary-only experiments and larger graphs."
        )
    )

    parser.add_argument("--graph", choices=["regular", "er", "ba"], required=True)
    parser.add_argument("--n", type=int, required=True)

    parser.add_argument("--k", type=int, default=4, help="Degree for regular graph.")
    parser.add_argument("--p", type=float, default=None, help="ER probability. If omitted, p=4/(n-1).")
    parser.add_argument("--m", type=int, default=2, help="m for BA graph.")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init", choices=["zeros", "ones", "random"], default="zeros")
    parser.add_argument("--init-p", type=float, default=0.30)

    parser.add_argument("--method", choices=["brd", "rm", "fp", "all"], default="all")

    parser.add_argument("--brd-iters", type=int, default=1000)
    parser.add_argument("--rm-iters", type=int, default=1000)
    parser.add_argument("--fp-iters", type=int, default=1000)

    # FP mode:
    # Keep sequential FP as the default because this is the version mainly used
    # in the current experiments. Use --fp-sync to force synchronous FP.
    parser.add_argument("--fp-sequential", dest="fp_sequential", action="store_true")
    parser.add_argument("--fp-sync", dest="fp_sequential", action="store_false")
    parser.set_defaults(fp_sequential=True)

    parser.add_argument("--rm-sequential", action="store_true", default=False)

    # CLI performance options for RM/FP.
    # Streamlit keeps full histories for visualization. The CLI defaults to
    # summary-only mode to reduce memory usage on large graphs.
    parser.add_argument(
        "--store-history",
        action="store_true",
        default=False,
        help=(
            "Store full per-iteration action profiles for RM/FP. "
            "Use only for debugging or small graphs. By default, CLI runs in "
            "summary-only mode to reduce memory usage."
        ),
    )

    parser.add_argument(
        "--check-every",
        type=int,
        default=1,
        help=(
            "Check PNE every k iterations for RM/FP. "
            "Use 1 for exact checked detection. Larger values reduce validation "
            "overhead on large graphs but may detect convergence later."
        ),
    )

    parser.add_argument("--out-dir", type=str, default="outputs/task1_cli")

    args = parser.parse_args()

    if args.check_every < 1:
        raise ValueError("--check-every must be >= 1")

    G, graph_label = make_graph(args)

    if args.method == "all":
        methods = ["brd", "rm", "fp"]
    else:
        methods = [args.method]

    rows: List[Dict[str, Any]] = []

    print("=" * 80)
    print(f"Graph: {graph_label}")
    print(f"Nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")
    print(f"Init: {args.init}" + (f" p={args.init_p}" if args.init == "random" else ""))
    print(f"Methods: {methods}")
    print(
        "CLI mode: "
        f"store_history={args.store_history}, "
        f"check_every={args.check_every}"
    )
    print("=" * 80)

    for method in methods:
        print(f"\nRunning {method.upper()} ...")
        row = run_one_method(args, G, graph_label, method)
        rows.append(row)

        print(
            f"{row['method']}: "
            f"PNE={row['reached_pne']} | "
            f"iter={row['iterations']} | "
            f"|S|={row['final_size_S']} | "
            f"minimal={row['is_minimal_security_set']} | "
            f"time={row['runtime_sec']:.2f}s"
        )

    graph_safe = safe_name(graph_label)
    init_safe = f"init-{args.init}" + (
        f"_p{str(args.init_p).replace('.', 'p')}" if args.init == "random" else ""
    )

    out_name = (
        f"task1_cli_{graph_safe}_{init_safe}_"
        f"BRD{args.brd_iters}_RM{args.rm_iters}_FP{args.fp_iters}_"
        f"check{args.check_every}_"
        f"hist{int(args.store_history)}_"
        f"seed{args.seed}.csv"
    )

    out_path = Path(args.out_dir) / out_name
    save_csv(rows, out_path)

    print("\nSaved CSV:")
    print(out_path)


if __name__ == "__main__":
    main()