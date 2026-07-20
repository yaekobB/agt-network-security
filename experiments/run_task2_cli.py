from __future__ import annotations

"""
experiments/run_task2_cli.py

Task 2-only CLI runner for the AGT Network Security project.

Purpose:
    This script runs only the coalitional-game part of the project:

        characteristic function v(C)
        -> Monte Carlo Shapley approximation
        -> Shapley ranking
        -> build security set
        -> prune to minimality
        -> export Task 2-specific CSV outputs

Why separate from run_task12_compare_cli.py?
    run_task12_compare_cli.py is for comparing Task 1 and Task 2 in one table.
    This file is for Task 2-only experiments and exports additional Task 2 outputs,
    such as Shapley rankings, build logs, and prune logs.

Recommended use:
    Use this file when you want to analyze v1/v2/v3 behavior independently.
    Use run_task12_compare_cli.py only when you want Task 1 vs Task 2 comparison.
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List

import networkx as nx

# ---------------------------------------------------------------------
# Make project imports work when this script is executed from project root.
# Example:
#   python experiments/run_task2_cli.py --graph regular --n 100 --k 4
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graphs.generators import GraphSpec, generate_graph
from src.task2_coalitional.characteristic import (
    CharParams,
    get_characteristic,
    validate_char_params,
)
from src.task2_coalitional.shapley import approximate_shapley
from src.task2_coalitional.induce_set import induce_minimal_security_set_from_ranking


def parse_list_argument(value: str) -> List[str]:
    """
    Parse comma-separated CLI lists.

    Example:
        "v1,v2,v3" -> ["v1", "v2", "v3"]
    """
    return [x.strip().lower() for x in value.split(",") if x.strip()]


def unique_keep_order(items: List[str]) -> List[str]:
    """
    Remove duplicates while preserving order.

    Example:
        ["v1", "v2", "v1"] -> ["v1", "v2"]

    This avoids running the same characteristic function twice by mistake.
    """
    seen = set()
    out = []

    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)

    return out


def safe_name(text: str) -> str:
    """
    Make a string safe for filenames.

    This is used to build output CSV names from graph labels and parameters.
    """
    return (
        str(text)
        .replace(" ", "")
        .replace("(", "_")
        .replace(")", "")
        .replace(",", "_")
        .replace("=", "")
        .replace(".", "p")
        .replace("/", "_")
        .replace("|", "_")
    )


def validate_args(args) -> None:
    """
    Validate CLI inputs before running expensive Shapley computations.

    This prevents unclear NetworkX errors and protects against invalid
    characteristic-function parameters.
    """

    if args.n < 2:
        raise ValueError("--n must be at least 2.")

    # Graph-specific validation.
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

    # Shapley sampling validation.
    if args.K < 1:
        raise ValueError("--K must be >= 1.")

    # Characteristic-function parameter validation.
    params = CharParams(alpha=float(args.alpha), gamma=float(args.gamma))
    validate_char_params(params)

    # Characteristic-function selection validation.
    chars = unique_keep_order(parse_list_argument(args.chars))
    if not chars:
        raise ValueError("--chars must contain at least one characteristic function.")

    allowed = {"v1", "v2", "v3"}
    invalid = [c for c in chars if c not in allowed]

    if invalid:
        raise ValueError(
            f"Invalid characteristic function(s): {invalid}. "
            f"Allowed values: {sorted(allowed)}"
        )


def make_graph(args) -> tuple[nx.Graph, str]:
    """
    Generate the graph used for Task 2.

    Supported graph types:
        regular:
            random k-regular graph

        er:
            Erdős-Rényi graph.
            If --p is omitted, we use p = 4/(n-1), giving expected degree ≈ 4.

        ba:
            Barabási-Albert graph with parameter m.
    """

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

    # The generator already normalizes labels, but this keeps the CLI robust
    # if a future graph generator returns non-integer labels.
    try:
        G = nx.relabel_nodes(G, {v: int(v) for v in G.nodes()})
    except Exception:
        pass

    return G, label


def save_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    """
    Save rows to CSV.

    The field order follows the first row. This is simple and predictable
    because each output table has a fixed row structure.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError("No rows to save.")

    fieldnames = list(rows[0].keys())

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_one_characteristic(
    *,
    G: nx.Graph,
    graph_label: str,
    cname: str,
    params: CharParams,
    K: int,
    seed: int,
    prune: bool,
    log_intermediate_minimality: bool,
) -> tuple[
    Dict[str, Any],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    """
    Run the full Task 2 pipeline for one characteristic function.

    Pipeline:
        1. Select characteristic function v(C).
        2. Approximate Shapley values by Monte Carlo permutations.
        3. Rank nodes by descending Shapley value.
        4. Add nodes in ranking order until a security set is reached.
        5. Prune redundant nodes to obtain an inclusion-wise minimal NSS.
        6. Return summary, ranking, build log, and prune log rows.

    Important:
        Shapley values provide the ranking.
        Final minimality is enforced by the pruning step, not by Shapley alone.
    """

    v_fn = get_characteristic(cname)

    # Monte Carlo Shapley approximation.
    shap = approximate_shapley(
        G,
        v_fn,
        params,
        samples=int(K),
        seed=int(seed),
    )

    # Build and prune the Shapley-induced security set.
    #
    # In CLI mode, log_intermediate_minimality is False by default.
    # This skips expensive minimality checks at intermediate build/prune steps.
    # Final security and final minimality are still always checked inside
    # induce_minimal_security_set_from_ranking(...).
    induced = induce_minimal_security_set_from_ranking(
        G,
        shap.ranking,
        prune=bool(prune),
        log_intermediate_minimality=bool(log_intermediate_minimality),
    )

    build_size = (
        int(induced.reached_security_at_size)
        if induced.reached_security_at_size is not None
        else None
    )

    final_size = int(induced.size_S)

    pruned_count = (
        max(0, int(build_size) - final_size)
        if build_size is not None
        else None
    )

    # -----------------------------------------------------------------
    # Summary row:
    # One row per characteristic function.
    # This is the main CSV table for Task 2 experiments.
    # -----------------------------------------------------------------
    summary_row = {
        "seed": int(seed),
        "graph": graph_label,
        "n": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "task": "Task 2",
        "approach": "Coalitional game",
        "characteristic": cname,
        "method": f"Shapley({cname})",
        "K": int(K),
        "alpha": float(params.alpha),
        "gamma": float(params.gamma),
        "prune": bool(prune),
        "log_intermediate_minimality": bool(log_intermediate_minimality),
        "build_size": build_size,
        "final_size": final_size,
        "size_ratio": final_size / max(G.number_of_nodes(), 1),
        "pruned_count": pruned_count,
        "final_security": bool(induced.final_is_security_set),
        "final_minimal": bool(induced.final_is_minimal),
        "satisfied_frac": float(induced.satisfied_frac),
        "secured_edges_frac": float(induced.secured_edges_frac),
        "shapley_runtime_ms": float(shap.runtime_ms),
        "shapley_runtime_sec": float(shap.runtime_ms) / 1000.0,
        "cache_size": int(shap.cache_size),
        "cache_hits": int(shap.cache_hits),
    }

    # -----------------------------------------------------------------
    # Ranking rows:
    # Full Shapley ranking. Useful for debugging and report examples.
    # Export only when --export-ranking is used.
    # -----------------------------------------------------------------
    ranking_rows = []
    for rank, node in enumerate(shap.ranking, start=1):
        ranking_rows.append({
            "seed": int(seed),
            "graph": graph_label,
            "characteristic": cname,
            "rank": rank,
            "node": node,
            "phi": float(shap.phi[node]),
        })

    # -----------------------------------------------------------------
    # Build rows:
    # Shows how the Shapley ranking gradually builds a security set.
    # Export only when --export-build-prune is used.
    # -----------------------------------------------------------------
    build_rows = []
    for s in induced.steps:
        build_rows.append({
            "seed": int(seed),
            "graph": graph_label,
            "characteristic": cname,
            "phase": "build",
            "t": s.t,
            "added_node": s.added_node,
            "size_S": s.size_S,
            "is_security_set": s.is_security_set,
            "is_minimal": s.is_minimal,
        })

    # -----------------------------------------------------------------
    # Prune rows:
    # Shows which nodes were tested/removed after the build phase.
    # Export only when --export-build-prune is used.
    # -----------------------------------------------------------------
    prune_rows = []
    for p in induced.prune_steps:
        prune_rows.append({
            "seed": int(seed),
            "graph": graph_label,
            "characteristic": cname,
            "phase": "prune",
            "step": p.step,
            "candidate_removed": p.candidate_removed,
            "removed": p.removed,
            "size_S_after": p.size_S_after,
            "still_security_set": p.still_security_set,
            "is_minimal_after": p.is_minimal_after,
        })

    return summary_row, ranking_rows, build_rows, prune_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Task 2 coalitional-game Shapley experiments only."
    )

    # -----------------------------------------------------------------
    # Graph options
    # -----------------------------------------------------------------
    parser.add_argument("--graph", choices=["regular", "er", "ba"], required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, default=4, help="Degree for regular graph.")
    parser.add_argument("--p", type=float, default=None, help="ER probability. If omitted, p=4/(n-1).")
    parser.add_argument("--m", type=int, default=2, help="m for BA graph.")

    # -----------------------------------------------------------------
    # Task 2 options
    # -----------------------------------------------------------------
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--chars",
        type=str,
        default="v1,v2,v3",
        help="Comma-separated characteristic functions to run, e.g. v1,v2,v3.",
    )

    parser.add_argument(
        "--K",
        type=int,
        default=500,
        help="Monte Carlo Shapley samples/permutations.",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.70,
        help="Weight on node satisfaction in v2/v3. alpha=0.7 is the report default.",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Size-efficiency strength used by v3.",
    )

    parser.add_argument(
        "--no-prune",
        action="store_true",
        default=False,
        help=(
            "Disable pruning. Normally you should NOT use this for final results, "
            "because Task 2 requires a minimal NSS."
        ),
    )

    parser.add_argument(
        "--log-intermediate-minimality",
        action="store_true",
        default=False,
        help=(
            "Compute minimality at intermediate build/prune log steps. "
            "Useful for debugging or UI-style explanation. "
            "Disabled by default for faster CLI experiments. "
            "Final minimality is always checked regardless of this flag."
        ),
    )

    # -----------------------------------------------------------------
    # Optional exports
    # -----------------------------------------------------------------
    parser.add_argument(
        "--export-ranking",
        action="store_true",
        default=False,
        help="Export full Shapley ranking CSV.",
    )

    parser.add_argument(
        "--export-build-prune",
        action="store_true",
        default=False,
        help="Export build and prune logs CSV.",
    )

    # -----------------------------------------------------------------
    # Output directory
    # -----------------------------------------------------------------
    parser.add_argument("--out-dir", type=str, default="outputs/task2_cli")

    args = parser.parse_args()
    validate_args(args)

    chars = unique_keep_order(parse_list_argument(args.chars))

    params = CharParams(alpha=float(args.alpha), gamma=float(args.gamma))
    validate_char_params(params)

    G, graph_label = make_graph(args)

    # Pruning is enabled by default because the final Task 2 goal is a minimal NSS.
    prune = not bool(args.no_prune)

    # CLI default is optimized summary mode.
    # Use --log-intermediate-minimality only when you specifically want detailed
    # intermediate minimality values in build/prune logs.
    log_intermediate_minimality = bool(args.log_intermediate_minimality)

    print("=" * 80)
    print("Task 2 CLI — Coalitional Game + Shapley")
    print("=" * 80)
    print(f"Graph: {graph_label}")
    print(f"Nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")
    print(f"Seed: {args.seed}")
    print(f"Characteristics: {chars}")
    print(f"K={args.K}, alpha={args.alpha}, gamma={args.gamma}")
    print(f"prune={prune}")
    print(f"log_intermediate_minimality={log_intermediate_minimality}")
    print("=" * 80)

    summary_rows: List[Dict[str, Any]] = []
    all_ranking_rows: List[Dict[str, Any]] = []
    all_build_rows: List[Dict[str, Any]] = []
    all_prune_rows: List[Dict[str, Any]] = []

    for cname in chars:
        print(f"\nRunning Shapley({cname}) ...")

        summary, ranking_rows, build_rows, prune_rows = run_one_characteristic(
            G=G,
            graph_label=graph_label,
            cname=cname,
            params=params,
            K=int(args.K),
            seed=int(args.seed),
            prune=prune,
            log_intermediate_minimality=log_intermediate_minimality,
        )

        summary_rows.append(summary)
        all_ranking_rows.extend(ranking_rows)
        all_build_rows.extend(build_rows)
        all_prune_rows.extend(prune_rows)

        print(
            f"Shapley({cname}): "
            f"build_size={summary['build_size']} | "
            f"final_size={summary['final_size']} | "
            f"minimal={summary['final_minimal']} | "
            f"security={summary['final_security']} | "
            f"time={summary['shapley_runtime_sec']:.2f}s"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    graph_safe = safe_name(graph_label)
    chars_safe = "chars-" + "-".join(chars)
    alpha_safe = str(args.alpha).replace(".", "p")
    gamma_safe = str(args.gamma).replace(".", "p")

    base_name = (
        f"task2_{graph_safe}_{chars_safe}_"
        f"K{args.K}_alpha{alpha_safe}_gamma{gamma_safe}_"
        f"prune{int(prune)}_seed{args.seed}"
    )

    # -----------------------------------------------------------------
    # Always save the summary table.
    # -----------------------------------------------------------------
    summary_path = out_dir / f"{base_name}_summary.csv"
    save_csv(summary_rows, summary_path)

    print("\nSaved summary CSV:")
    print(summary_path)

    # -----------------------------------------------------------------
    # Optional detailed exports.
    # -----------------------------------------------------------------
    if args.export_ranking:
        ranking_path = out_dir / f"{base_name}_ranking.csv"
        save_csv(all_ranking_rows, ranking_path)
        print("\nSaved ranking CSV:")
        print(ranking_path)

    if args.export_build_prune:
        build_path = out_dir / f"{base_name}_build.csv"
        save_csv(all_build_rows, build_path)
        print("\nSaved build log CSV:")
        print(build_path)

        prune_path = out_dir / f"{base_name}_prune.csv"

        # Prune rows can be empty if the build set was already minimal.
        if all_prune_rows:
            save_csv(all_prune_rows, prune_path)
            print("\nSaved prune log CSV:")
            print(prune_path)
        else:
            print("\nNo prune rows to save.")

    print("\nDone.")


if __name__ == "__main__":
    main()