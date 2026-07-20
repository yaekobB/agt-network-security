from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graphs.generators import GraphSpec, generate_graph
from src.security.checks import (
    is_security_set,
    is_minimal_security_set,
    coverage_stats,
)
from src.task1_strategic.game import (
    default_params,
    set_from_actions,
    is_pure_nash_equilibrium,
)
from src.task1_strategic.dynamics import init_actions, brd_run
from src.task1_strategic.regret_matching import regret_matching_run
from src.task1_strategic.fictitious_play import fictitious_play_run

from src.task2_coalitional.characteristic import CharParams
from src.task2_coalitional.compare_task1_task2 import run_task2_variant


# Fixed Task 1 game constants.
# These are modeling constants, not experiment parameters.
TASK1_COST = 1.0
TASK1_PENALTY = 100.0


def parse_list_argument(value: str) -> List[str]:
    """Parse comma-separated CLI lists such as 'brd,rm,fp'."""
    return [x.strip().lower() for x in value.split(",") if x.strip()]


def validate_args(args) -> None:
    """Validate CLI arguments before running expensive experiments."""

    if args.n < 2:
        raise ValueError("--n must be at least 2.")

    if not (0.0 <= args.init_p <= 1.0):
        raise ValueError("--init-p must be in [0, 1].")

    if args.check_every < 1:
        raise ValueError("--check-every must be >= 1.")

    if args.K < 1:
        raise ValueError("--K must be >= 1.")

    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError("--alpha must be in [0, 1].")

    if args.gamma < 0:
        raise ValueError("--gamma must be non-negative.")

    if args.graph == "regular":
        if args.k < 0:
            raise ValueError("--k must be non-negative for regular graphs.")
        if args.k >= args.n:
            raise ValueError("--k must be smaller than --n for regular graphs.")
        if (args.n * args.k) % 2 != 0:
            raise ValueError("For regular graphs, n*k must be even.")

    elif args.graph == "er":
        if args.p is not None and not (0.0 <= args.p <= 1.0):
            raise ValueError("--p must be in [0, 1] for Erdős-Rényi graphs.")

    elif args.graph == "ba":
        if args.m < 1:
            raise ValueError("--m must be at least 1 for Barabási-Albert graphs.")
        if args.m >= args.n:
            raise ValueError("--m must be smaller than --n for Barabási-Albert graphs.")

    task1_methods = parse_list_argument(args.task1_methods)
    task2_chars = parse_list_argument(args.task2_chars)

    if not task1_methods and not task2_chars:
        raise ValueError("At least one Task 1 method or Task 2 characteristic must be provided.")

    allowed_task1 = {"brd", "rm", "fp"}
    invalid_task1 = [m for m in task1_methods if m not in allowed_task1]
    if invalid_task1:
        raise ValueError(
            f"Invalid Task 1 method(s): {invalid_task1}. "
            f"Allowed values: {sorted(allowed_task1)}"
        )

    allowed_task2 = {"v1", "v2", "v3"}
    invalid_task2 = [c for c in task2_chars if c not in allowed_task2]
    if invalid_task2:
        raise ValueError(
            f"Invalid Task 2 characteristic(s): {invalid_task2}. "
            f"Allowed values: {sorted(allowed_task2)}"
        )


def make_graph(args) -> tuple[nx.Graph, str]:
    """Generate one graph instance used by both Task 1 and Task 2."""

    if args.graph == "regular":
        spec = GraphSpec(type="regular", n=args.n, k=args.k, seed=args.seed)
        label = f"regular(n={args.n},k={args.k})"

    elif args.graph == "er":
        p = float(args.p) if args.p is not None else 4.0 / max(args.n - 1, 1)
        spec = GraphSpec(type="erdos_renyi", n=args.n, p=p, seed=args.seed)
        label = f"erdos_renyi(n={args.n},p={p:.6f})"

    elif args.graph == "ba":
        spec = GraphSpec(type="barabasi_albert", n=args.n, m=args.m, seed=args.seed)
        label = f"barabasi_albert(n={args.n},m={args.m})"

    else:
        raise ValueError(f"Unknown graph type: {args.graph}")

    G = generate_graph(spec)

    # Keep integer node labels when possible for cleaner CSVs.
    try:
        G = nx.relabel_nodes(G, {v: int(v) for v in G.nodes()})
    except Exception:
        pass

    return G, label


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


def task1_summary_row(
    *,
    method: str,
    G: nx.Graph,
    params,
    result,
    runtime_ms: float,
    graph_label: str,
    seed: int,
    init_mode: str,
    init_p: Optional[float],
    max_iter: int,
    fp_mode: Optional[str],
    store_history: bool,
    check_every: int,
) -> Dict[str, Any]:
    """Convert one Task 1 result into the common Task 1 vs Task 2 comparison format."""

    S = set_from_actions(result.final_actions)
    cov = coverage_stats(G, S)

    return {
        "seed": seed,
        "graph": graph_label,
        "n": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "init_mode": init_mode,
        "init_p": init_p,
        "approach": "Task 1",
        "method": method,
        "size_S": len(S),
        "size_ratio": len(S) / max(G.number_of_nodes(), 1),
        "reached_pne": bool(is_pure_nash_equilibrium(G, result.final_actions, params)),
        "iterations": int(result.iterations),
        "is_security_set": bool(is_security_set(G, S)),
        "is_minimal_security_set": bool(is_minimal_security_set(G, S)),
        "satisfied_frac": cov.fraction_satisfied_nodes,
        "secured_edges_frac": cov.fraction_secured_edges,
        "runtime_ms": runtime_ms,
        "runtime_sec": runtime_ms / 1000.0,
        "max_iter": max_iter,
        "fp_mode": fp_mode,
        "K": None,
        "alpha": None,
        "gamma": None,
        "store_history": store_history,
        "check_every": check_every,
    }


def run_task1_method(
    *,
    args,
    G: nx.Graph,
    graph_label: str,
    params,
    a0,
    init_p_used: Optional[float],
    method: str,
) -> Dict[str, Any]:
    """Run one Task 1 method in CLI summary mode."""

    method = method.lower().strip()
    t0 = time.perf_counter()

    if method == "brd":
        result = brd_run(
            G,
            dict(a0),
            params,
            max_iters=args.brd_iters,
            seed=args.seed,
            shuffle_each_round=True,
        )
        method_name = "BRD"
        max_iter = args.brd_iters
        fp_mode = None

    elif method == "rm":
        result = regret_matching_run(
            G,
            dict(a0),
            params,
            T=args.rm_iters,
            seed=args.seed,
            sequential=args.rm_sequential,
            shuffle_each_round=True,
            store_history=args.store_history,
            check_every=args.check_every,
        )
        method_name = "RM-seq" if args.rm_sequential else "RM"
        max_iter = args.rm_iters
        fp_mode = None

    elif method == "fp":
        result = fictitious_play_run(
            G,
            dict(a0),
            params,
            T=args.fp_iters,
            seed=args.seed,
            sequential=args.fp_sequential,
            shuffle_each_round=True,
            store_history=args.store_history,
            check_every=args.check_every,
        )
        method_name = "FP-seq" if args.fp_sequential else "FP-sync"
        max_iter = args.fp_iters
        fp_mode = "sequential" if args.fp_sequential else "synchronous"

    else:
        raise ValueError(f"Unknown Task 1 method: {method}")

    runtime_ms = (time.perf_counter() - t0) * 1000.0

    return task1_summary_row(
        method=method_name,
        G=G,
        params=params,
        result=result,
        runtime_ms=runtime_ms,
        graph_label=graph_label,
        seed=args.seed,
        init_mode=args.init,
        init_p=init_p_used,
        max_iter=max_iter,
        fp_mode=fp_mode,
        store_history=args.store_history,
        check_every=args.check_every,
    )


def run_task2_method(
    *,
    args,
    G: nx.Graph,
    graph_label: str,
    params2: CharParams,
    char_name: str,
) -> Dict[str, Any]:
    """Run one Task 2 Shapley variant and convert it to the common comparison format."""

    crow = run_task2_variant(
        G,
        char_name,
        params2,
        K=int(args.K),
        seed=int(args.seed),
    )

    return {
        "seed": args.seed,
        "graph": graph_label,
        "n": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "init_mode": args.init,
        "init_p": float(args.init_p) if args.init == "random" else None,
        "approach": "Task 2",
        "method": crow.method,
        "size_S": int(crow.size_S),
        "size_ratio": int(crow.size_S) / max(G.number_of_nodes(), 1),
        "reached_pne": None,
        "iterations": None,
        "is_security_set": bool(crow.is_security_set),
        "is_minimal_security_set": bool(crow.is_minimal_security_set),
        "satisfied_frac": float(crow.satisfied_frac),
        "secured_edges_frac": float(crow.secured_edges_frac),
        "runtime_ms": float(crow.runtime_ms),
        "runtime_sec": float(crow.runtime_ms) / 1000.0,
        "max_iter": None,
        "fp_mode": None,
        "K": int(args.K),
        "alpha": float(args.alpha),
        "gamma": float(args.gamma),
        "store_history": None,
        "check_every": None,
    }


def save_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError("No rows to save.")

    fieldnames = list(rows[0].keys())

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Task 1 vs Task 2 comparison outside Streamlit."
    )

    # Graph options
    parser.add_argument("--graph", choices=["regular", "er", "ba"], required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, default=4, help="Degree for regular graph.")
    parser.add_argument("--p", type=float, default=None, help="ER probability. If omitted, p=4/(n-1).")
    parser.add_argument("--m", type=int, default=2, help="m for BA graph.")

    # Shared experiment options
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init", choices=["zeros", "ones", "random"], default="zeros")
    parser.add_argument("--init-p", type=float, default=0.30)

    # Task 1 options
    parser.add_argument("--task1-methods", type=str, default="brd,rm,fp")
    parser.add_argument("--brd-iters", type=int, default=1000)
    parser.add_argument("--rm-iters", type=int, default=1000)
    parser.add_argument("--fp-iters", type=int, default=1000)

    parser.add_argument("--fp-sequential", dest="fp_sequential", action="store_true")
    parser.add_argument("--fp-sync", dest="fp_sequential", action="store_false")
    parser.set_defaults(fp_sequential=True)

    parser.add_argument("--rm-sequential", action="store_true", default=False)

    # CLI performance options for RM/FP.
    parser.add_argument(
        "--store-history",
        action="store_true",
        default=False,
        help="Store full RM/FP histories. Use only for debugging or small graphs.",
    )

    parser.add_argument(
        "--check-every",
        type=int,
        default=1,
        help="Check PNE every k iterations for RM/FP. Use 1 for exact checked detection.",
    )

    # Task 2 options
    parser.add_argument("--task2-chars", type=str, default="v1,v2,v3")
    parser.add_argument("--K", type=int, default=500, help="Monte Carlo Shapley samples.")

    # Default alpha is 0.70 to stay consistent with the main project/report setting.
    parser.add_argument("--alpha", type=float, default=0.70)
    parser.add_argument("--gamma", type=float, default=1.0)

    # Output
    parser.add_argument("--out-dir", type=str, default="outputs/task12_compare_cli")

    args = parser.parse_args()
    validate_args(args)

    G, graph_label = make_graph(args)

    task1_methods = parse_list_argument(args.task1_methods)
    task2_chars = parse_list_argument(args.task2_chars)

    # Fixed Task 1 parameters, created once and shared by BRD/RM/FP.
    params = default_params(G, c=TASK1_COST, penalty=TASK1_PENALTY)

    # Same initial profile for all Task 1 methods.
    if args.init == "random":
        a0 = init_actions(G, "random", p=float(args.init_p), seed=args.seed)
        init_p_used = float(args.init_p)
    else:
        a0 = init_actions(G, args.init, seed=args.seed)
        init_p_used = None

    # Shared Task 2 parameters for all characteristic functions.
    params2 = CharParams(alpha=float(args.alpha), gamma=float(args.gamma))

    rows: List[Dict[str, Any]] = []

    print("=" * 80)
    print("Task 1 vs Task 2 CLI comparison")
    print(f"Graph: {graph_label}")
    print(f"Nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")
    print(f"Seed: {args.seed}")
    print(f"Init: {args.init}" + (f" p={args.init_p}" if args.init == "random" else ""))
    print(f"Task 1 methods: {task1_methods}")
    print(f"Task 2 chars: {task2_chars}")
    print(f"Task 1 params: cost={TASK1_COST}, penalty={TASK1_PENALTY}")
    print(f"Task 2 params: K={args.K}, alpha={args.alpha}, gamma={args.gamma}")
    print(f"CLI mode: store_history={args.store_history}, check_every={args.check_every}")
    print("=" * 80)

    # Run Task 1 methods.
    for method in task1_methods:
        print(f"\nRunning Task 1 {method.upper()} ...")
        row = run_task1_method(
            args=args,
            G=G,
            graph_label=graph_label,
            params=params,
            a0=a0,
            init_p_used=init_p_used,
            method=method,
        )
        rows.append(row)

        print(
            f"{row['method']}: "
            f"PNE={row['reached_pne']} | "
            f"iter={row['iterations']} | "
            f"|S|={row['size_S']} | "
            f"minimal={row['is_minimal_security_set']} | "
            f"time={row['runtime_sec']:.2f}s"
        )

    # Run Task 2 Shapley variants.
    for cname in task2_chars:
        print(f"\nRunning Task 2 Shapley({cname}) ...")
        row = run_task2_method(
            args=args,
            G=G,
            graph_label=graph_label,
            params2=params2,
            char_name=cname,
        )
        rows.append(row)

        print(
            f"{row['method']}: "
            f"|S|={row['size_S']} | "
            f"security={row['is_security_set']} | "
            f"minimal={row['is_minimal_security_set']} | "
            f"time={row['runtime_sec']:.2f}s"
        )

    graph_safe = safe_name(graph_label)
    init_safe = f"init-{args.init}" + (
        f"_p{str(args.init_p).replace('.', 'p')}" if args.init == "random" else ""
    )

    out_name = (
        f"task12_compare_{graph_safe}_{init_safe}_"
        f"K{args.K}_alpha{str(args.alpha).replace('.', 'p')}_"
        f"gamma{str(args.gamma).replace('.', 'p')}_"
        f"check{args.check_every}_hist{int(args.store_history)}_"
        f"seed{args.seed}.csv"
    )

    out_path = Path(args.out_dir) / out_name
    save_csv(rows, out_path)

    print("\nSaved CSV:")
    print(out_path)


if __name__ == "__main__":
    main()