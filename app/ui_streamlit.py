"""
app/ui_streamlit.py

Streamlit UI for the AGT Network Security project.

Features:
- Generate graphs (regular / ER / BA) using the same config parameters
- Visualize the graph live for small/medium instances
- Disable graph drawing for large experiments to reduce memory/time usage
- Highlight a chosen node-set S (empty / all / random percentage / Task 1 result)
- Compute and display security-set checks + coverage stats
- Save/download figure for report/presentation when visualization is enabled
"""

from __future__ import annotations
import io

import sys
from pathlib import Path
import random
import pandas as pd

import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
import time

# Ensure project root is on PYTHONPATH so "import src.*" works when running Streamlit.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.task1_strategic.game import default_params, set_from_actions
from src.task1_strategic.dynamics import init_actions, brd_run

from src.graphs.generators import GraphSpec, generate_graph
from src.security.checks import is_security_set, is_minimal_security_set, coverage_stats
from src.utils.seed import set_seed
from src.task1_strategic.regret_matching import regret_matching_run
from src.task1_strategic.game import is_pure_nash_equilibrium
from src.task1_strategic.fictitious_play import fictitious_play_run
from src.task1_strategic.compare import run_compare_task1

# -- Task 2 imports ---
from src.task2_coalitional.characteristic import CharParams, get_characteristic, coalition_stats, score_components
from src.task2_coalitional.shapley import approximate_shapley
from src.task2_coalitional.induce_set import induce_minimal_security_set_from_ranking
from src.task2_coalitional.compare_task1_task2 import run_task2_variant
from src.task2_coalitional.experiments import CompareConfig, run_task2_characteristic_compare, run_task2_preset_sweep

# -- Task 3 imports ---
# from src.task3_planner.planner import PlannerConfig, run_task3_planner

# -- Task 3 imports ---
from src.task3_planner.planner import (
    PlannerConfig,
    Vendor,
    run_task3_planner,
    solve_infinite_items,
    solve_limited_items,
)

# -- Task 4 imports ---
from src.task4_auction.auction import AuctionConfig, run_task4_auction


def draw_graph(G: nx.Graph, S: set[int], title: str) -> plt.Figure:
    """
    Draw the graph with nodes in S highlighted.

    Notes:
    - Uses spring_layout with a fixed seed for consistent small-graph visualization.
    - Labels are drawn only for small graphs because labels are expensive and unreadable
      for larger graphs.
    - Large-graph drawing is controlled from the UI visualization policy.
    """
    fig = plt.figure(figsize=(7, 5))
    ax = plt.gca()
    ax.set_title(title)

    pos = nx.spring_layout(G, seed=42)

    nodes = list(G.nodes())
    in_S = [v for v in nodes if v in S]
    out_S = [v for v in nodes if v not in S]

    # Draw edges first
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.4)

    # Draw nodes not in S
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=out_S,
        ax=ax,
        node_size=450,
        node_color="#D9D9D9",
        edgecolors="#555555",
        linewidths=0.8,
    )

    # Draw nodes in S
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=in_S,
        ax=ax,
        node_size=600,
        node_color="#2ECC71",
        edgecolors="#145A32",
        linewidths=1.2,
    )

    # Labels are useful for small graphs but expensive/unreadable for larger graphs.
    if G.number_of_nodes() <= 100:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=9)

    ax.axis("off")
    return fig


def save_figure(fig: plt.Figure, filename: str) -> Path:
    out_dir = Path("outputs") / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    return out_path


def fig_to_png_bytes(fig: plt.Figure) -> bytes:
    """Convert a Matplotlib figure into PNG bytes for Streamlit download button."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=200)
    buf.seek(0)
    return buf.read()


def main() -> None:
    st.set_page_config(page_title="AGT Network Security UI", layout="wide")
    st.title("AGT 2025/26 — Network Security Sets")

    st.info(
        "Revised version: Task 1 separates BRD, Fictitious Play, and Regret Matching; "
        "Task 2 uses non-negative characteristic functions; "
        "Task 3 uses normalized non-negative vendor utilities."
    )

    # -----------------------------
    # 0) GLOBAL SIDEBAR
    # -----------------------------
    with st.sidebar:
        st.header("Graph Generator")

        seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=42, step=1)
        set_seed(int(seed))

        gtype = st.selectbox(
            "Graph type",
            ["regular", "erdos_renyi", "barabasi_albert"],
            index=0,
        )

        if gtype == "regular":
            n = st.slider("n (nodes)", 5, 100000, 1000, 1)
            k = st.slider("k (degree)", 0, min(20, n - 1), 4, 1)
            st.caption("Note: regular graphs require n*k to be even.")
            spec = GraphSpec(type="regular", n=int(n), k=int(k), seed=int(seed))

        elif gtype == "erdos_renyi":
            n = st.slider("n (nodes)", 5, 100000, 1000, 1)

            er_mode = st.selectbox(
                "ER p mode",
                [
                    "auto_sparse_degree_4",
                    "manual_p",
                    "density_test",
                ],
                index=0,
                help=(
                    "auto_sparse_degree_4 sets p ≈ 4/(n-1), comparable to regular k=4 "
                    "and BA m=2. Use manual_p for custom experiments."
                ),
            )

            if er_mode == "auto_sparse_degree_4":
                p = 4.0 / max(int(n) - 1, 1)
                st.caption(
                    f"Using p = {p:.6f}, giving expected average degree ≈ {p * (int(n) - 1):.2f}"
                )

            elif er_mode == "density_test":
                expected_degree = st.selectbox(
                    "Expected average degree",
                    [2, 4, 8, 16, 32],
                    index=1,
                    help="p is computed as expected_degree / (n-1).",
                )
                p = float(expected_degree) / max(int(n) - 1, 1)
                st.caption(
                    f"Using p = {p:.6f}, expected average degree ≈ {expected_degree}"
                )

            else:
                p = st.number_input(
                    "p (edge probability)",
                    min_value=0.0,
                    max_value=1.0,
                    value=4.0 / max(int(n) - 1, 1),
                    step=0.0001,
                    format="%.6f",
                    help="For sparse ER graphs, use small p values such as 0.004, 0.0008, or 0.0004.",
                )
                st.caption(
                    f"Expected average degree ≈ {float(p) * (int(n) - 1):.2f}"
                )

            spec = GraphSpec(type="erdos_renyi", n=int(n), p=float(p), seed=int(seed))

        else:
            n = st.slider("n (nodes)", 5, 100000, 1000, 1)
            m = st.slider("m (edges per new node)", 1, min(10, n - 1), 2, 1)
            spec = GraphSpec(type="barabasi_albert", n=int(n), m=int(m), seed=int(seed))

        st.divider()
        st.header("Export")
        save_png = st.checkbox("Save figure to outputs/figures/", value=False)

    # -----------------------------
    # 1) BUILD GRAPH ONCE
    # -----------------------------
    try:
        G = generate_graph(spec)
    except Exception as e:
        st.error(f"Graph generation failed: {e}")
        return

    # Ensure integer node labels when possible.
    try:
        G = nx.relabel_nodes(G, {v: int(v) for v in G.nodes()})
    except Exception:
        pass

    # Task 1 params shared by later tabs
    params = default_params(G, c=1.0, penalty=100.0)

    # -----------------------------
    # 1.5) VISUALIZATION POLICY
    # -----------------------------
    VIS_SMALL_N = 100
    VIS_MAX_N = 500

    enable_graph_viz = st.sidebar.checkbox(
        "Enable graph visualization",
        value=(G.number_of_nodes() <= VIS_SMALL_N),
        help=(
            "Recommended only for small/medium graphs. "
            "For large experiments, disable visualization and use metrics/CSV outputs."
        ),
    )

    if G.number_of_nodes() > VIS_MAX_N:
        enable_graph_viz = False
        st.sidebar.info(
            f"Graph visualization disabled because n={G.number_of_nodes()} > {VIS_MAX_N}. "
            "Security checks and experiment metrics will still run."
        )

    # -----------------------------
    # 2) TABS
    # -----------------------------
    tab0, tab1, tab2, tab3, tab4, tabR = st.tabs([
        "Task 0 — Visualize & Check",
        "Task 1 — Strategic Game (BRD/RM/FP)",
        "Task 2 — Coalitional + Shapley ",
        "Task 3 — Planner Assignment",
        "Task 4 — Secure Path Auction",
        "Report / Export",
    ])

    # =========================================================
    # TAB 0: Task 0 — manual S selection + checks + stats
    # =========================================================
    with tab0:
        st.subheader("Task 0: Select/Highlight S and verify security properties")

        st.info(
            "Task 0 generates the graph, lets you choose a candidate set S, "
            "and checks whether S is a Network Security Set and whether it is minimal. "
            "For large graphs, visualization can be disabled while checks and statistics still run."
        )

        # --- (A) Selection mode ---
        mode = st.selectbox(
            "Selection mode",
            ["empty", "all", "random_percent", "task1_equilibrium"],
            index=2,
        )

        rand_pct = 0.30
        if mode == "random_percent":
            rand_pct = st.slider("Random selection %", 0.0, 1.0, 0.30, 0.05)

        # --- (B) Build S ---
        nodes = list(G.nodes())

        if mode == "empty":
            S = set()

        elif mode == "all":
            S = set(nodes)

        elif mode == "random_percent":
            random.seed(int(seed))
            S = {v for v in nodes if random.random() < float(rand_pct)}

        else:
            # If user selects equilibrium here, show latest Task 1 result if it exists.
            res = st.session_state.get("task1_result", None)
            if res is None:
                S = set()
                st.warning("No Task 1 result yet. Go to Task 1 tab and run dynamics.")
            else:
                S = set_from_actions(res.final_actions)

        # --- (C) Security checks and statistics ---
        ok = is_security_set(G, S)
        minimal = is_minimal_security_set(G, S)
        stats = coverage_stats(G, S)

        left, right = st.columns([2, 1], gap="large")

        # --- (D) Visualization / export ---
        with left:
            filename = f"graph_{spec.type}_S{len(S)}.png"

            if enable_graph_viz:
                fig = draw_graph(G, S, title=f"{spec.type} graph | |S|={len(S)}")
                png_bytes = fig_to_png_bytes(fig)

                st.pyplot(fig, clear_figure=True)

                if save_png:
                    out_dir = Path("outputs") / "figures"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / filename
                    out_path.write_bytes(png_bytes)
                    st.success(f"Saved figure: {out_path.as_posix()}")

                st.download_button(
                    label="⬇️ Download PNG",
                    data=png_bytes,
                    file_name=filename,
                    mime="image/png",
                    width="stretch",
                )

                plt.close(fig)

            else:
                st.info(
                    "Graph visualization is disabled for this run. "
                    "Security checks and coverage statistics are still computed."
                )
                st.caption(
                    f"Graph summary: type={spec.type}, "
                    f"nodes={G.number_of_nodes()}, edges={G.number_of_edges()}, |S|={len(S)}"
                )

        # --- (E) Security checks + coverage stats ---
        with right:
            st.subheader("Security-Set Checks")
            st.write(f"**is_security_set:** {ok}")
            st.write(f"**is_minimal_security_set:** {minimal}")

            st.subheader("Coverage Stats")
            st.write(
                {
                    "nodes": stats.n_nodes,
                    "edges": stats.n_edges,
                    "size(S)": stats.size_S,
                    "satisfied_nodes": (
                        f"{stats.satisfied_nodes}/{stats.n_nodes} "
                        f"({stats.fraction_satisfied_nodes:.2%})"
                    ),
                    "secured_edges": (
                        f"{stats.secured_edges}/{stats.n_edges} "
                        f"({stats.fraction_secured_edges:.2%})"
                    ),
                }
            )

            st.caption(
                "Minimal means inclusion-wise minimal: removing any node from S breaks "
                "the security-set property. It does not necessarily mean minimum-cardinality."
            )
    # =========================================================
    # TAB 1: Task 1 — run BRD/RM/FP + replay iterations
    # =========================================================
    with tab1:
        st.subheader("Task 1: Strategic Game — BRD, Fictitious Play, and Regret Matching")

        st.info(
            "BRD responds to the current action profile. "
            "Fictitious Play responds to empirical beliefs built from historical play. "
            "Regret Matching samples actions according to cumulative positive regrets. "
            "PNE checking is used only for validation/stopping, not as part of the update rule."
        )

        # ---------------------------------------------------------
        # Experiment / visualization policy for Task 1
        # ---------------------------------------------------------
        experiment_mode = st.checkbox(
            "Experiment mode: reduce memory usage",
            value=(G.number_of_nodes() > 500),
            help=(
                "Recommended for large graphs or long runs. "
                "Disables iteration replay visualization and focuses on metrics/CSV outputs."
            ),
        )

        if experiment_mode:
            enable_task1_graph_viz = False
            st.info(
                "Experiment mode is ON. Iteration graph replay is disabled. "
                "Final metrics, convergence plots, and CSV exports remain available."
            )
        else:
            enable_task1_graph_viz = enable_graph_viz

        # --- (A) Controls ---
        dynamics_method = st.selectbox(
            "Dynamics method",
            ["Best Response Dynamics (BRD)", "Regret Matching (RM)", "Fictitious Play (FP)"],
            index=0,
        )

        init_mode = st.selectbox("Init profile", ["zeros", "ones", "random"], index=0)

        init_p = 0.30
        if init_mode == "random":
            init_p = st.slider("Init random p", 0.0, 1.0, 0.30, 0.05)

        rm_T = 1000
        if dynamics_method.startswith("Regret"):
            rm_T = st.slider("RM iterations (T)", 10, 10000, 1000, 10)

        fp_T = 1000
        fp_sequential = False

        if dynamics_method.startswith("Fictitious"):
            fp_T = st.slider("FP iterations (T)", 10, 10000, 1000, 10)

            fp_sequential = st.checkbox(
                "Use sequential FP",
                value=True,
                help=(
                    "If checked, FP updates players one by one. "
                    "If unchecked, FP updates all players synchronously."
                ),
            )

            st.caption(
                "FP mode: sequential update" if fp_sequential
                else "FP mode: synchronous update"
            )

        run_dyn = st.button("▶ Run Task 1 dynamics")

        # --- (B) Build initial profile a0 ---
        if init_mode == "zeros":
            a0 = init_actions(G, "zeros", seed=int(seed))
        elif init_mode == "ones":
            a0 = init_actions(G, "ones", seed=int(seed))
        else:
            a0 = init_actions(G, "random", p=float(init_p), seed=int(seed))

        # --- (C) Run selected dynamics ---
        if run_dyn:
            if dynamics_method.startswith("Best Response"):
                res = brd_run(
                    G,
                    a0,
                    params,
                    max_iters=500,
                    shuffle_each_round=True,
                    seed=int(seed),
                )
                st.session_state["task1_method"] = "BRD"

            elif dynamics_method.startswith("Fictitious"):
                res = fictitious_play_run(
                    G,
                    a0,
                    params,
                    T=int(fp_T),
                    seed=int(seed),
                    sequential=bool(fp_sequential),
                )

                st.session_state["task1_method"] = "FP-seq" if fp_sequential else "FP-sync"

            else:
                res = regret_matching_run(
                    G,
                    a0,
                    params,
                    T=int(rm_T),
                    seed=int(seed),
                )
                st.session_state["task1_method"] = "RM"

            st.session_state["task1_result"] = res

        # --- (D) Display result + optional replay ---
        res = st.session_state.get("task1_result", None)
        method = st.session_state.get("task1_method", "Dynamics")

        if res is None:
            st.info("Run a dynamics method to compute an equilibrium/security set.")
        else:
            pne = is_pure_nash_equilibrium(G, res.final_actions, params)

            if pne:
                st.success(f"{method} finished: reached PNE ✅ | iterations={res.iterations}")
            else:
                st.warning(f"{method} finished: NOT PNE ⚠️ | iterations={res.iterations}")

            st.caption(f"PNE check: {pne}")

            # ---------------------------------------------------------
            # Replay selected iteration only in demo mode
            # ---------------------------------------------------------
            if (
                (not experiment_mode)
                and hasattr(res, "history_actions")
                and res.history_actions
            ):
                t = st.slider(
                    f"{method} iteration to display",
                    min_value=0,
                    max_value=len(res.history_actions) - 1,
                    value=len(res.history_actions) - 1,
                    step=1,
                )

                a_t = res.history_actions[t]
                S = {i for i, ai in a_t.items() if ai == 1}

                if t == 0:
                    changes_t = "NA"
                else:
                    if (
                        hasattr(res, "history_changed")
                        and len(res.history_changed) == len(res.history_actions)
                    ):
                        changes_t = res.history_changed[t]      # FP/RM aligned format
                    else:
                        changes_t = res.history_changed[t - 1]  # BRD shifted format

                st.caption(
                    f"Showing {method} iteration {t}/{len(res.history_actions)-1} | "
                    f"|S|={len(S)} | changes={changes_t}"
                )

            else:
                S = set_from_actions(res.final_actions)

                if experiment_mode:
                    st.caption(
                        "Iteration replay is disabled in experiment mode. "
                        "Showing final action profile only."
                    )
                elif not hasattr(res, "history_actions") or not res.history_actions:
                    st.caption("No iteration history available. Showing final action profile only.")

            # ---------------------------------------------------------
            # Security checks and final/selected S statistics
            # ---------------------------------------------------------
            ok = is_security_set(G, S)
            minimal = is_minimal_security_set(G, S)
            stats = coverage_stats(G, S)

            left, right = st.columns([2, 1], gap="large")

            with left:
                filename = f"task1_{method}_{spec.type}_S{len(S)}.png"

                if enable_task1_graph_viz:
                    fig = draw_graph(G, S, title=f"{spec.type} graph | |S|={len(S)}")
                    png_bytes = fig_to_png_bytes(fig)

                    st.pyplot(fig, clear_figure=True)

                    if save_png:
                        out_dir = Path("outputs") / "figures"
                        out_dir.mkdir(parents=True, exist_ok=True)
                        out_path = out_dir / filename
                        out_path.write_bytes(png_bytes)
                        st.success(f"Saved figure: {out_path.as_posix()}")

                    st.download_button(
                        label="⬇️ Download PNG",
                        data=png_bytes,
                        file_name=filename,
                        mime="image/png",
                        width="stretch",
                    )

                    plt.close(fig)

                else:
                    st.info(
                        "Task 1 graph visualization is disabled for this run. "
                        "Security checks and metrics are still computed."
                    )
                    st.caption(
                        f"Final/selected profile summary: method={method}, "
                        f"graph={spec.type}, nodes={G.number_of_nodes()}, "
                        f"edges={G.number_of_edges()}, |S|={len(S)}"
                    )

            with right:
                st.subheader("Security-Set Checks")
                st.write(f"**is_security_set:** {ok}")
                st.write(f"**is_minimal_security_set:** {minimal}")

                st.subheader("Coverage Stats")
                st.write(
                    {
                        "nodes": stats.n_nodes,
                        "edges": stats.n_edges,
                        "size(S)": stats.size_S,
                        "satisfied_nodes": (
                            f"{stats.satisfied_nodes}/{stats.n_nodes} "
                            f"({stats.fraction_satisfied_nodes:.2%})"
                        ),
                        "secured_edges": (
                            f"{stats.secured_edges}/{stats.n_edges} "
                            f"({stats.fraction_secured_edges:.2%})"
                        ),
                    }
                )

                st.caption(
                    "Minimal means inclusion-wise minimal: removing any node from S breaks "
                    "the security-set property. It does not necessarily mean minimum-cardinality."
                )

        # =========================================================
        # Compare BRD vs RM vs FP
        # =========================================================
        st.divider()
        st.subheader("Compare BRD vs RM vs FP")

        with st.expander("Comparison settings (BRD vs RM vs FP)", expanded=True):
            brd_max_iters = st.slider("BRD max passes", 0, 10000, 1000, 100)
            rm_T_cmp = st.slider("RM iterations (T)", 0, 10000, 1000, 100, key="task1_cmp_rm_T")
            fp_T_cmp = st.slider("FP iterations (T)", 0, 10000, 1000, 100, key="task1_cmp_fp_T")

            fp_sequential_cmp = st.checkbox(
                "Use sequential FP in comparison",
                value=True,
                key="task1_cmp_fp_sequential",
                help=(
                    "If checked, the FP row in BRD/RM/FP comparison uses sequential=True. "
                    "If unchecked, FP uses synchronous updates."
                ),
            )

            st.caption(
                "Comparison FP mode: sequential" if fp_sequential_cmp
                else "Comparison FP mode: synchronous"
            )

            show_full_logs = st.checkbox(
                "Display full per-iteration log table",
                value=(G.number_of_nodes() <= 500),
                help=(
                    "For large experiments, leave this unchecked and use CSV download instead."
                ),
            )

        colA, colB, colC = st.columns([1, 1, 2])
        with colA:
            run_all = st.button("⚡ Run ALL (BRD + RM + FP)")
        with colB:
            save_logs = st.checkbox("Save logs to outputs/logs/", value=False)

        # Helper to describe graph parameters
        def graph_param_str(gtype: str, n: int, k=None, p=None, m=None) -> str:
            if gtype == "regular":
                return f"regular(n={n}, k={k})"
            if gtype == "erdos_renyi":
                return f"erdos_renyi(n={n}, p={p:.2f})"
            return f"barabasi_albert(n={n}, m={m})"

        if run_all:
            cmp_res = run_compare_task1(
            G,
            a0,
            params,
            seed=int(seed),
            brd_max_iters=int(brd_max_iters),
            rm_T=int(rm_T_cmp),
            fp_T=int(fp_T_cmp),
            rm_kwargs={},
            fp_kwargs={"sequential": bool(fp_sequential_cmp)},
            brd_kwargs={},
        )

            gparam = graph_param_str(
                gtype,
                int(n),
                k=k if gtype == "regular" else None,
                p=p if gtype == "erdos_renyi" else None,
                m=m if gtype == "barabasi_albert" else None,
            )

            cmp_res.meta = {
                "seed": int(seed),
                "graph": gparam,
                "init_mode": init_mode,
                "init_p": float(init_p) if init_mode == "random" else None,
                "brd_max_iters": int(brd_max_iters),
                "rm_T": int(rm_T_cmp),
                "fp_T": int(fp_T_cmp),
                "fp_mode": "sequential" if fp_sequential_cmp else "synchronous",
            }

            st.session_state["task1_compare"] = cmp_res

        cmp_res = st.session_state.get("task1_compare", None)

        if cmp_res is not None:
            meta = getattr(cmp_res, "meta", {}) or {}

            seed_used = meta.get("seed", int(seed))
            gparam_used = meta.get("graph", "unknown-graph")
            init_mode_used = meta.get("init_mode", init_mode)
            init_p_used = meta.get("init_p", None)

            brd_max_used = meta.get("brd_max_iters", None)
            rm_T_used = meta.get("rm_T", None)
            fp_T_used = meta.get("fp_T", None)
            
            fp_mode_used = meta.get("fp_mode", "unknown")

            df_sum = pd.DataFrame(cmp_res.summaries)

            df_sum.insert(0, "seed", seed_used)
            df_sum.insert(1, "graph", gparam_used)
            df_sum.insert(2, "init_mode", init_mode_used)
            df_sum.insert(3, "init_p", init_p_used)
            df_sum.insert(4, "fp_mode", fp_mode_used)
            
            st.markdown("### Summary of comparison results")
            
            fp_mode_used = meta.get("fp_mode", "unknown")

            st.caption(
                f"Context: {gparam_used} | seed={seed_used} | init={init_mode_used}"
                + (
                    f" (p={init_p_used:.2f})"
                    if init_mode_used == "random" and init_p_used is not None
                    else ""
                )
                + (
                    f" | BRD_max={brd_max_used} RM_T={rm_T_used} FP_T={fp_T_used}"
                    if brd_max_used is not None
                    else ""
                )
                + f" | FP mode={fp_mode_used}"
            )

            st.dataframe(df_sum, width="stretch")
            
            # -----------------------------
            # Download summary table as CSV
            # -----------------------------
            def safe_name(x) -> str:
                return (
                    str(x)
                    .replace(" ", "")
                    .replace("(", "_")
                    .replace(")", "")
                    .replace(",", "_")
                    .replace("=", "")
                    .replace(".", "p")
                    .replace("|", "_")
                    .replace("/", "_")
                )

            safe_graph = safe_name(gparam_used)

            safe_init = f"init-{init_mode_used}"
            if init_mode_used == "random" and init_p_used is not None:
                safe_init += f"_p{float(init_p_used):.2f}".replace(".", "p")

            fp_mode_used = meta.get("fp_mode", "unknown")

            summary_fname = (
                f"task1_summary_"
                f"{safe_graph}_"
                f"{safe_init}_"
                f"BRD{brd_max_used}_"
                f"RM{rm_T_used}_"
                f"FP{fp_T_used}_"
                f"seed{seed_used}_"
                f"FP-{fp_mode_used}.csv"
            )

            summary_csv = df_sum.to_csv(index=False).encode("utf-8")

            st.download_button(
                "⬇️ Download summary table (CSV)",
                data=summary_csv,
                file_name=summary_fname,
                mime="text/csv",
                width="stretch",
                key="download_task1_summary_csv",
            )

            # Helper: convert logs to aligned dataframe for charts
            def to_series_df(field: str) -> pd.DataFrame:
                series = {}
                max_t = 0

                for m_name, rows in cmp_res.logs.items():
                    d = {r["t"]: r.get(field, None) for r in rows}
                    if d:
                        max_t = max(max_t, max(d.keys()))
                    series[m_name] = d

                data = {"t": list(range(max_t + 1))}
                for m_name, d in series.items():
                    data[m_name] = [d.get(t, None) for t in data["t"]]

                return pd.DataFrame(data).set_index("t")
            
            

            st.markdown("### Convergence plots")
            st.line_chart(to_series_df("size_S"))
            st.line_chart(to_series_df("changed"))

            # Build logs dataframe only when needed for display/download/save
            with st.expander("Per-iteration logs (table + CSV)"):
                df_logs = pd.concat(
                    [pd.DataFrame(rows) for rows in cmp_res.logs.values()],
                    ignore_index=True,
                ).sort_values(["method", "t"])

                df_logs["seed"] = seed_used
                df_logs["graph"] = gparam_used
                df_logs["init_mode"] = init_mode_used
                df_logs["init_p"] = init_p_used
                df_logs["fp_mode"] = fp_mode_used

                if show_full_logs:
                    st.dataframe(df_logs, width="stretch")
                else:
                    st.caption(
                        "Full log table display is disabled to reduce UI memory usage. "
                        "You can still download or save the CSV."
                    )

                csv_bytes = df_logs.to_csv(index=False).encode("utf-8")

                safe_graph = (
                    str(gparam_used)
                    .replace("(", "_")
                    .replace(")", "")
                    .replace(",", "_")
                    .replace("=", "")
                    .replace(" ", "")
                )
                safe_init = f"{init_mode_used}" + (
                    f"_p{init_p_used:.2f}"
                    if init_mode_used == "random" and init_p_used is not None
                    else ""
                )
                fname = f"task1_compare_{safe_graph}_{safe_init}_seed{seed_used}.csv"

                st.download_button(
                    "⬇️ Download comparison logs (CSV)",
                    data=csv_bytes,
                    file_name=fname,
                    mime="text/csv",
                    width="stretch",
                )

                if save_logs:
                    out_dir = Path("outputs") / "logs"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / fname
                    out_path.write_bytes(csv_bytes)
                    st.success(f"Saved: {out_path.as_posix()}")
    
    # =========================================================
    # TAB 2: Task 2 — Coalitional + Shapley
    # =========================================================

    with tab2:
        st.subheader("Task 2: Coalitional game + Shapley values")

        char_name = st.selectbox(
            "Characteristic function v(C)",
            [
                "v1 — node satisfaction",
                "v2 — node + edge security",
                "v3 — size-efficient security",
            ],
            index=1,
        )

        st.info(
            "All characteristic functions are non-negative security scores. "
            "v1 measures the fraction of nodes satisfying the NSS rule. "
            "v2 combines node satisfaction with secured-edge coverage. "
            "v3 multiplies v2 by a size-efficiency factor, so larger coalitions are discouraged "
            "without producing negative coalition values."
        )

        alpha = st.slider(
            "alpha (node satisfaction vs edge coverage)",
            0.0, 1.0, 0.7, 0.05
        )

        gamma = st.slider(
            "gamma (size-efficiency strength for v3)",
            0.0, 5.0, 1.0, 0.1
        )

        K = st.slider("Monte Carlo samples (permutations)", 50, 5000, 500, 50)

        run_task2 = st.button("▶ Run Task 2 Shapley")

        if run_task2:
            params2 = CharParams(alpha=float(alpha), gamma=float(gamma))
            v_key = char_name.split()[0]  # "v1" / "v2" / "v3"
            v_fn = get_characteristic(v_key)

            gparam = graph_param_str(
                gtype, int(n),
                k=k if gtype == "regular" else None,
                p=p if gtype == "erdos_renyi" else None,
                m=m if gtype == "barabasi_albert" else None,
            )

            shap = approximate_shapley(G, v_fn, params2, samples=int(K), seed=int(seed))
            induced = induce_minimal_security_set_from_ranking(G, shap.ranking, prune=True)

            st.session_state["task2_shapley"] = shap
            st.session_state["task2_induced"] = induced
            st.session_state["task2_meta"] = {
                "seed": int(seed),
                "graph": gparam,
                "char": char_name,
                "v_key": v_key,
                "alpha": float(alpha),
                "gamma": float(gamma),
                "K": int(K),
            }

        shap = st.session_state.get("task2_shapley")
        induced = st.session_state.get("task2_induced")
        meta = st.session_state.get("task2_meta")

        if shap and induced and meta:
            if "gamma" not in meta:
                st.warning(
                    "Old Task 2 session state detected. Please rerun Task 2 Shapley "
                    "so the new gamma-based characteristic functions are used."
                )
            else:
                # -----------------------------
                # (1) Shapley ranking table
                # -----------------------------
                st.markdown("### Shapley ranking")
                df_phi = (
                    pd.DataFrame([{"node": k, "phi": v} for k, v in shap.phi.items()])
                    .sort_values("phi", ascending=False)
                )
                st.dataframe(df_phi, width="stretch")

                st.caption(
                    f"Runtime: {shap.runtime_ms:.1f} ms | "
                    f"cache_size={shap.cache_size} | cache_hits={shap.cache_hits}"
                )

                # -----------------------------
                # (2) Build vs Prune
                # -----------------------------
                st.markdown("### Induced minimal security set from Shapley")

                reached_at = induced.reached_security_at_size
                final_size = induced.size_S

                st.caption(
                    f"Context: {meta['graph']} | seed={meta['seed']} | char={meta['char']} | "
                    f"alpha={meta['alpha']:.2f}, gamma={meta['gamma']:.2f}, K={meta['K']}"
                )

                if reached_at is not None:
                    st.write(f"Build phase reached a security set at **|S| = {reached_at}**")
                else:
                    st.write("Build phase did not reach a security set. This is unexpected if the ranking covers all nodes.")

                st.write(f"Final after pruning: **|S| = {final_size}**")
                st.write(f"is_security_set = {induced.final_is_security_set}")
                st.write(f"is_minimal_security_set = {induced.final_is_minimal}")

                st.caption(
                    f"Coverage: satisfied_nodes={induced.satisfied_frac:.3f}, "
                    f"secured_edges={induced.secured_edges_frac:.3f}"
                )

                st.markdown("**Final S (ordered by Shapley ranking):**")
                st.code(str(induced.S_ordered))

                # -----------------------------
                # (2.5) Coalition scoring snapshot
                # -----------------------------
                st.markdown("### Coalition scoring snapshot")

                C_empty = set()

                reached_step = getattr(induced, "reached_security_at_step", None)
                if reached_step is not None:
                    C_build = set(getattr(induced, "S_build", induced.S))
                else:
                    C_build = set(induced.S)

                C_final = set(induced.S)

                v_key = meta.get("v_key", meta["char"].split()[0])
                v_fn = get_characteristic(v_key)
                params2 = CharParams(
                    alpha=float(meta["alpha"]),
                    gamma=float(meta["gamma"]),
                )

                def comp_row(name: str, C: set):
                    stC = coalition_stats(G, C)
                    comps = score_components(G, C, params2)

                    return {
                        "coalition": name,
                        "|C|": stC.size_C,
                        "is_security_set": stC.is_security_set,
                        "is_minimal": stC.is_minimal_security_set,
                        "node_satisfaction": round(comps["node_satisfaction"], 4),
                        "edge_coverage": round(comps["edge_coverage"], 4),
                        "combined_security": round(comps["combined_security"], 4),
                        "violations_frac": round(comps["violations_frac"], 4),
                        "size_frac": round(comps["size_frac"], 4),
                        "size_efficiency": round(comps["size_efficiency"], 4),
                        "v(C)": round(v_fn(G, C, params2), 4),
                    }

                rows = [
                    comp_row("Empty coalition", C_empty),
                    comp_row("Build-reached security set", C_build),
                    comp_row("Final after pruning", C_final),
                ]

                df_comp = pd.DataFrame(rows)
                st.dataframe(df_comp, width="stretch")

                st.caption(
                    "Interpretation: v(C) is a non-negative security score. "
                    "Node satisfaction and edge coverage increase coalition value. "
                    "For v3, size_efficiency rewards compact coalitions through a multiplicative factor, "
                    "not a subtractive penalty. Shapley ranks nodes by average marginal contribution."
                )

                # -----------------------------
                # (3) Build log
                # -----------------------------
                with st.expander("Build log (steps) + CSV"):
                    df_build = pd.DataFrame([{
                        "t": s.t,
                        "added_node": s.added_node,
                        "size_S": s.size_S,
                        "is_security_set": s.is_security_set,
                        "is_minimal": s.is_minimal,
                    } for s in induced.steps])

                    for k_meta, v_meta in meta.items():
                        df_build[k_meta] = v_meta

                    st.dataframe(df_build, width="stretch")

                    csv_build = df_build.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download Task2 build log (CSV)",
                        data=csv_build,
                        file_name=f"task2_build_{meta['v_key']}_{meta['graph'].replace(' ', '')}_seed{meta['seed']}.csv",
                        mime="text/csv",
                        width="stretch",
                    )

                # -----------------------------
                # (4) Prune log
                # -----------------------------
                with st.expander("Prune log (removals) + CSV"):
                    df_prune = pd.DataFrame([{
                        "step": p.step,
                        "candidate_to_remove": p.candidate_removed,
                        "removed": p.removed,
                        "size_S_after": p.size_S_after,
                        "trial_security_set": p.still_security_set,
                        "current_is_minimal_after": p.is_minimal_after,
                    } for p in getattr(induced, "prune_steps", [])])

                    for k_meta, v_meta in meta.items():
                        df_prune[k_meta] = v_meta

                    st.dataframe(df_prune, width="stretch")

                    csv_prune = df_prune.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download Task2 prune log (CSV)",
                        data=csv_prune,
                        file_name=f"task2_prune_{meta['v_key']}_{meta['graph'].replace(' ', '')}_seed{meta['seed']}.csv",
                        mime="text/csv",
                        width="stretch",
                    )

                # -----------------------------
                # (5) Combined export
                # -----------------------------
                with st.expander("Combined export (build + prune)"):
                    df_build2 = df_build.copy()
                    df_build2.insert(0, "phase", "build")

                    df_prune2 = df_prune.copy()
                    if not df_prune2.empty:
                        df_prune2.insert(0, "phase", "prune")

                    df_all = pd.concat([df_build2, df_prune2], ignore_index=True, sort=False)

                    st.dataframe(df_all, width="stretch")

                    csv_all = df_all.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download Task2 (build+prune) combined CSV",
                        data=csv_all,
                        file_name=f"task2_build_prune_{meta['v_key']}_{meta['graph'].replace(' ', '')}_seed{meta['seed']}.csv",
                        mime="text/csv",
                        width="stretch",
                    )

        # -----------------------------
        # (6) Compare Task2 variants
        # -----------------------------
        with st.expander("✅ Compare v1 vs v2 vs v3 (final |S|, runtime, stability)"):

            seed_start = st.number_input("Seed start", value=42, step=1, key="t2_seed_start")
            num_seeds = st.slider("Number of seeds", 3, 50, 10, 1, key="t2_num_seeds")
            seeds = list(range(int(seed_start), int(seed_start) + int(num_seeds)))

            K_cmp = st.slider("K (permutations) for stability", 50, 5000, 500, 50, key="t2_K_cmp")

            alpha_cmp = st.slider("alpha", 0.0, 1.0, 0.7, 0.05, key="t2_alpha_cmp")
            gamma_cmp = st.slider(
                "gamma (size-efficiency strength for v3)",
                0.0, 5.0, 1.0, 0.1,
                key="t2_gamma_cmp",
            )

            _graph_cache: dict[int, nx.Graph] = {}

            def graph_factory(s: int) -> nx.Graph:
                if s in _graph_cache:
                    return _graph_cache[s].copy()

                spec_s = GraphSpec(
                    type=spec.type,
                    n=int(spec.n),
                    k=int(spec.k) if spec.k is not None else None,
                    p=float(spec.p) if spec.p is not None else None,
                    m=int(spec.m) if spec.m is not None else None,
                    seed=int(s),
                )
                Gs = generate_graph(spec_s)

                try:
                    Gs = nx.relabel_nodes(Gs, {v: int(v) for v in Gs.nodes()})
                except Exception:
                    pass

                _graph_cache[s] = Gs
                return Gs.copy()

            meta_cmp = {
                "graph_type": gtype,
                "n": int(n),
                "k": int(k) if gtype == "regular" else None,
                "p": float(p) if gtype == "erdos_renyi" else None,
                "m": int(m) if gtype == "barabasi_albert" else None,
                "seed_start": int(seed_start),
                "num_seeds": int(num_seeds),
                "K": int(K_cmp),
            }

            col1, col2 = st.columns(2)
            with col1:
                run_stab = st.button("▶ Run stability comparison (v1/v2/v3)", key="t2_run_stab")
            with col2:
                run_presets = st.button(
                    "⚡ Run PRESET sweep (baseline / strong_size_eff / node_focus / edge_focus)",
                    key="t2_run_presets",
                )

            # -------------------------------------------------------
            # (A) Single config run: v1/v2/v3
            # -------------------------------------------------------
            if run_stab:
                cfg = CompareConfig(
                    char_names=["v1", "v2", "v3"],
                    K=int(K_cmp),
                    alpha=float(alpha_cmp),
                    gamma=float(gamma_cmp),
                    prune=True,
                )

                df_runs, df_summary = run_task2_characteristic_compare(
                    graph_factory,
                    seeds,
                    cfg,
                    extra_meta={
                        **meta_cmp,
                        "mode": "single",
                        "alpha": float(alpha_cmp),
                        "gamma": float(gamma_cmp),
                    },
                )

                st.session_state["task2_stab_runs"] = df_runs
                st.session_state["task2_stab_summary"] = df_summary

            # -------------------------------------------------------
            # (B) Preset sweep
            # -------------------------------------------------------
            PRESETS = [
                {"preset": "baseline",        "alpha": 0.7, "gamma": 1.0},
                {"preset": "strong_size_eff", "alpha": 0.7, "gamma": 2.5},
                {"preset": "node_focus",      "alpha": 0.9, "gamma": 1.0},
                {"preset": "edge_focus",      "alpha": 0.3, "gamma": 1.0},
            ]

            if run_presets:
                df_runs, df_summary = run_task2_preset_sweep(
                    graph_factory=graph_factory,
                    seeds=seeds,
                    presets=PRESETS,
                    K=int(K_cmp),
                    char_names=["v1", "v2", "v3"],
                    prune=True,
                    extra_meta={**meta_cmp, "mode": "preset_sweep"},
                )

                st.session_state["task2_preset_runs"] = df_runs
                st.session_state["task2_preset_summary"] = df_summary

            # -------------------------------------------------------
            # DISPLAY: single-config results
            # -------------------------------------------------------
            df_runs = st.session_state.get("task2_stab_runs")
            df_summary = st.session_state.get("task2_stab_summary")

            if df_runs is not None and df_summary is not None:
                st.markdown("### Stability summary (mean/std over seeds) — single config")
                st.dataframe(df_summary, width="stretch")

                st.markdown("### All runs (seed × v) — single config")
                st.dataframe(df_runs, width="stretch")

                st.download_button(
                    "⬇️ Download stability runs CSV (single config)",
                    data=df_runs.to_csv(index=False).encode("utf-8"),
                    file_name=f"task2_stability_runs_{gtype}_seed{int(seed_start)}_n{int(num_seeds)}_single.csv",
                    mime="text/csv",
                    width="stretch",
                )

                st.download_button(
                    "⬇️ Download stability summary CSV (single config)",
                    data=df_summary.to_csv(index=False).encode("utf-8"),
                    file_name=f"task2_stability_summary_{gtype}_seed{int(seed_start)}_n{int(num_seeds)}_single.csv",
                    mime="text/csv",
                    width="stretch",
                )

            # -------------------------------------------------------
            # DISPLAY: preset-sweep results
            # -------------------------------------------------------
            df_pruns = st.session_state.get("task2_preset_runs")
            df_psum = st.session_state.get("task2_preset_summary")

            if df_pruns is not None and df_psum is not None:
                st.markdown("### Preset sweep summary (mean/std over seeds)")
                st.dataframe(df_psum, width="stretch")

                st.markdown("### Preset sweep — all runs (seed × v × preset)")
                st.dataframe(df_pruns, width="stretch")

                st.download_button(
                    "⬇️ Download preset sweep runs CSV",
                    data=df_pruns.to_csv(index=False).encode("utf-8"),
                    file_name=f"task2_preset_runs_{gtype}_seed{int(seed_start)}_n{int(num_seeds)}.csv",
                    mime="text/csv",
                    width="stretch",
                )

                st.download_button(
                    "⬇️ Download preset sweep summary CSV",
                    data=df_psum.to_csv(index=False).encode("utf-8"),
                    file_name=f"task2_preset_summary_{gtype}_seed{int(seed_start)}_n{int(num_seeds)}.csv",
                    mime="text/csv",
                    width="stretch",
                )

        # ---------------------------------------------
        # (7) Task 1 vs Task 2 comparison
        # ---------------------------------------------

        st.divider()
        st.subheader("Task 1 vs Task 2 Comparison")

        run_big = st.button("⚡ Run Comparison (Task1 + Task2)")

        if run_big:
            rows = []

            task1_cmp = run_compare_task1(
                G, a0, params,
                seed=int(seed),
                brd_max_iters=int(brd_max_iters),
                rm_T=int(rm_T),
                fp_T=int(fp_T),
            )

            for r in task1_cmp.summaries:
                rows.append(r)

            params2 = CharParams(alpha=float(alpha), gamma=float(gamma))

            for cname in ["v1", "v2", "v3"]:
                crow = run_task2_variant(G, cname, params2, K=int(K), seed=int(seed))
                rows.append({
                    "method": crow.method,
                    "size_S": crow.size_S,
                    "is_security_set": crow.is_security_set,
                    "is_minimal_security_set": crow.is_minimal_security_set,
                    "satisfied_frac": crow.satisfied_frac,
                    "secured_edges_frac": crow.secured_edges_frac,
                    "runtime_ms": crow.runtime_ms,
                    **crow.extra,
                })

            df_cmp = pd.DataFrame(rows)

            gparam = graph_param_str(
                gtype, int(n),
                k=k if gtype == "regular" else None,
                p=p if gtype == "erdos_renyi" else None,
                m=m if gtype == "barabasi_albert" else None,
            )

            df_cmp.insert(0, "seed", int(seed))
            df_cmp.insert(1, "graph", gparam)
            df_cmp.insert(2, "init_mode", init_mode)
            df_cmp.insert(3, "init_p", float(init_p) if init_mode == "random" else None)

            st.session_state["task12_cmp_df"] = df_cmp

        df_cmp = st.session_state.get("task12_cmp_df")
        if df_cmp is not None:
            st.dataframe(df_cmp, width="stretch")
            csv = df_cmp.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download comparison (CSV)",
                data=csv,
                file_name=f"task12_compare_seed{int(seed)}.csv",
                mime="text/csv",
                width="stretch",
            )
    
    # =========================================================
    # TAB 3: Task 3 — Planner assignment
    # Inputs → Case A → Case B → Utilization → Why Unmatched → Summary
    # =========================================================
    with tab3:
        st.subheader("Task 3 — Planner Assignment (vendors ↔ players in S)")

        # ---- keep your width="stretch" approach with safe fallback ----
        def df_show(df):
            try:
                st.dataframe(df, width="stretch")
            except TypeError:
                st.dataframe(df, use_container_width=True)

        # ---------------------------------------------------------
        # 0) Choose S source
        # ---------------------------------------------------------
        S = None
        source = st.selectbox(
            "Choose S source",
            ["Task 2 (Shapley induced)", "Task 1 (strategic solution)"],
            index=0,
        )

        if source.startswith("Task 2"):
            induced = st.session_state.get("task2_induced")
            if induced is not None and hasattr(induced, "S"):
                S = sorted(list(induced.S))
        else:
            task1 = st.session_state.get("task1_result")
            if task1 is not None and hasattr(task1, "final_actions"):
                S = sorted([i for i, a in task1.final_actions.items() if a == 1])

        if not S:
            st.warning("No S found yet. Run Task 2 or Task 1 first to obtain a minimal security set S.")
        else:
            st.write(f"Using |S| = {len(S)} : {S}")

            # ---------------------------------------------------------
            # 1) Generator settings + Run button
            # ---------------------------------------------------------
            st.markdown("### Generator settings")

            st.info(
                "Task 3 assigns the selected security nodes in S to security-service vendors. "
                "For compatible pairs p_v ≤ b_i, the utility is "
                "u(i,v)=α(level/10)+(1−α)(1−price/budget). "
                "This utility is normalized and non-negative. "
                "Case A assumes infinite vendor supply. "
                "Case B uses limited vendor capacity and is solved as a welfare-maximizing min-cost flow assignment."
            )

            t3_seed = st.number_input(
                "Task 3 random seed",
                min_value=0,
                max_value=10_000,
                value=42,
                step=1,
            )

            n_vendors = st.slider("Number of vendors", 2, 30, 8, 1)

            utility_alpha = st.slider(
                "alpha (security-level weight)",
                0.0, 1.0, 0.7, 0.05,
                help=(
                    "alpha close to 1 gives more importance to vendor security level. "
                    "alpha close to 0 gives more importance to affordability."
                ),
            )

            cap_min = st.slider("Capacity min (limited case)", 1, 10, 1, 1)
            cap_max = st.slider("Capacity max (limited case)", cap_min, 10, max(3, cap_min), 1)

            balanced = st.checkbox("Balanced vendor generator (optional)", value=False)
            price_noise = 8.0
            if balanced:
                price_noise = st.slider("Balanced generator: price noise", 0.0, 30.0, 8.0, 1.0)

            run_t3 = st.button("▶ Run Task 3 Planner")

            if run_t3:
                cfg = PlannerConfig(
                    n_vendors=int(n_vendors),
                    alpha=float(utility_alpha),
                    cap_min=int(cap_min),
                    cap_max=int(cap_max),
                    seed=int(t3_seed),
                    balanced_vendors=bool(balanced),
                    price_noise=float(price_noise),
                )

                players, vendors, resA, resB = run_task3_planner(S, cfg)

                st.session_state["task3_players"] = players
                st.session_state["task3_vendors"] = vendors
                st.session_state["task3_resA"] = resA
                st.session_state["task3_resB"] = resB
                st.session_state["task3_cfg"] = cfg

            players = st.session_state.get("task3_players")
            vendors = st.session_state.get("task3_vendors")
            resA = st.session_state.get("task3_resA")
            resB = st.session_state.get("task3_resB")
            cfg = st.session_state.get("task3_cfg")

            if players and vendors and resA and resB and cfg:
                if not hasattr(cfg, "alpha"):
                    st.warning(
                        "Old Task 3 session state detected. Please rerun Task 3 Planner "
                        "so the new alpha-based utility is used."
                    )
                else:
                    # ---------------------------------------------------------
                    # 2) INPUTS
                    # ---------------------------------------------------------
                    st.markdown("## Inputs")

                    st.caption(
                        f"S source: {source} | |S|={len(S)} | "
                        f"cfg: seed={cfg.seed}, alpha={cfg.alpha:.2f}, "
                        f"n_vendors={cfg.n_vendors}, cap=[{cfg.cap_min}..{cfg.cap_max}]"
                    )

                    vendor_prices = [v.price for v in vendors]

                    df_players = pd.DataFrame([
                        {"player": p.node, "budget": p.budget}
                        for p in players
                    ])

                    df_players["affordable_vendors_count"] = df_players["budget"].apply(
                        lambda b: sum(1 for pr in vendor_prices if pr <= b)
                    )
                    df_players = df_players.sort_values(["budget", "player"], ascending=[True, True])

                    df_vendors = pd.DataFrame([
                        {
                            "vendor": v.vid,
                            "price": v.price,
                            "level": v.level,
                            "capacity": v.capacity,
                        }
                        for v in vendors
                    ]).sort_values(["price", "level"], ascending=[True, False])

                    with st.expander("Players (budgets + affordability)", expanded=False):
                        df_show(df_players)

                    with st.expander("Vendors (price / level / capacity)", expanded=False):
                        df_show(df_vendors)

                    max_budget = int(df_players["budget"].max()) if not df_players.empty else None
                    min_price = int(df_vendors["price"].min()) if not df_vendors.empty else None

                    unaffordable_vendors = []
                    if max_budget is not None:
                        unaffordable_vendors = df_vendors[df_vendors["price"] > max_budget]["vendor"].tolist()

                    cols = st.columns(3)
                    cols[0].metric("Max player budget", max_budget if max_budget is not None else "-")
                    cols[1].metric("Min vendor price", min_price if min_price is not None else "-")
                    cols[2].metric("Vendors unaffordable to all", len(unaffordable_vendors))

                    if unaffordable_vendors:
                        st.caption(
                            "Unaffordable for everyone (price > max budget): "
                            + ", ".join(unaffordable_vendors)
                        )

                    budget_of = {p.node: p.budget for p in players}

                    # ---------------------------------------------------------
                    # Helper: compute normalized utility table for explanation
                    # ---------------------------------------------------------
                    with st.expander("Compatible utility matrix explanation", expanded=False):
                        util_rows = []
                        for p in players:
                            for v in vendors:
                                if v.price <= p.budget:
                                    security_score = v.level / 10.0
                                    affordability_score = 1.0 - (v.price / p.budget)
                                    u = (
                                        cfg.alpha * security_score
                                        + (1.0 - cfg.alpha) * affordability_score
                                    )
                                    util_rows.append({
                                        "player": p.node,
                                        "budget": p.budget,
                                        "vendor": v.vid,
                                        "price": v.price,
                                        "level": v.level,
                                        "security_score": round(security_score, 4),
                                        "affordability_score": round(affordability_score, 4),
                                        "utility": round(u, 4),
                                    })

                        if util_rows:
                            df_util_matrix = pd.DataFrame(util_rows).sort_values(
                                ["player", "utility"], ascending=[True, False]
                            )
                            df_show(df_util_matrix)
                        else:
                            st.write("No compatible player-vendor pairs exist.")

                    # ---------------------------------------------------------
                    # 3) CASE A — Infinite items
                    # ---------------------------------------------------------
                    st.markdown("## Case A — Infinite items")
                    st.caption(
                        "Because vendor supply is unlimited, each player independently chooses "
                        "the compatible vendor with maximum utility."
                    )
                    st.caption(
                        f"Total welfare = {resA.total_welfare:.3f} | runtime = {resA.runtime_ms:.1f} ms | "
                        f"unmatched = {len(resA.unmatched)}"
                    )

                    df_A = pd.DataFrame([{
                        "player": i,
                        "budget": budget_of.get(i, None),
                        "vendor": resA.assignments[i],
                        "utility": round(resA.utilities[i], 4),
                    } for i in sorted(resA.assignments.keys())])
                    df_show(df_A)

                    with st.expander("Case A vendor load summary", expanded=False):
                        countsA = {}
                        for i, vid in resA.assignments.items():
                            countsA[vid] = countsA.get(vid, 0) + 1

                        df_loadA = (
                            pd.DataFrame([
                                {"vendor": k, "assigned_players": v}
                                for k, v in countsA.items()
                            ])
                            .sort_values("assigned_players", ascending=False)
                        )
                        df_show(df_loadA)

                    # ---------------------------------------------------------
                    # 4) CASE B — Limited items
                    # ---------------------------------------------------------
                    st.markdown("## Case B — Limited items (capacity)")
                    st.caption(
                        "With limited capacity, players compete for vendor slots. "
                        "Therefore, the assignment is solved globally as a maximum-welfare "
                        "assignment via min-cost flow."
                    )
                    st.caption(
                        f"Total welfare = {resB.total_welfare:.3f} | runtime = {resB.runtime_ms:.1f} ms | "
                        f"unmatched = {len(resB.unmatched)}"
                    )

                    df_B = pd.DataFrame([{
                        "player": i,
                        "budget": budget_of.get(i, None),
                        "vendor": resB.assignments[i],
                        "utility": round(resB.utilities[i], 4),
                    } for i in sorted(resB.assignments.keys())])
                    df_show(df_B)

                    # ---------------------------------------------------------
                    # 5) Vendor utilization (Case B)
                    # ---------------------------------------------------------
                    st.markdown("## Vendor utilization (Case B)")
                    loads = resB.vendor_loads or {}
                    util_rows = []

                    for v in vendors:
                        load = int(loads.get(v.vid, 0))
                        cap = int(v.capacity)
                        util = (100.0 * load / cap) if cap > 0 else 0.0

                        util_rows.append({
                            "vendor": v.vid,
                            "price": v.price,
                            "level": v.level,
                            "load": load,
                            "capacity": cap,
                            "utilization_%": round(util, 1),
                            "saturated": "YES" if load >= cap else "",
                        })

                    df_util = pd.DataFrame(util_rows).sort_values(
                        ["saturated", "utilization_%", "level", "price"],
                        ascending=[False, False, False, True],
                    )
                    df_show(df_util)

                    # ---------------------------------------------------------
                    # 6) Why unmatched? (Case B)
                    # ---------------------------------------------------------
                    st.markdown("## Why unmatched? (Case B)")

                    unmatched = sorted(list(resB.unmatched))
                    if not unmatched:
                        st.success("No unmatched players in Case B. Everyone was assigned within budget and capacity constraints.")
                    else:
                        saturated_set = set(df_util[df_util["saturated"] == "YES"]["vendor"].tolist())

                        def affordable_vids(b):
                            aff = [v.vid for v in vendors if v.price <= b]
                            return sorted(
                                aff,
                                key=lambda vid: next(x.price for x in vendors if x.vid == vid)
                            )

                        detail_rows = []
                        for i in unmatched:
                            b = budget_of.get(i, None)
                            aff = affordable_vids(b) if b is not None else []

                            if len(aff) == 0:
                                reason = "No affordable vendor (price > budget for all vendors)"
                            elif all(vid in saturated_set for vid in aff):
                                reason = "All affordable vendors are saturated (full capacity)"
                            else:
                                reason = "Affordable vendor exists but was not selected by the welfare-maximizing assignment"

                            detail_rows.append({
                                "player": i,
                                "budget": b,
                                "affordable_vendors_count": len(aff),
                                "affordable_vendors": ", ".join(aff) if aff else "(none)",
                                "reason": reason,
                            })

                        st.info("UNASSIGNED occurs when no budget-feasible vendor is available or compatible vendors are exhausted by capacity constraints.")
                        df_unm = pd.DataFrame(detail_rows)
                        df_show(df_unm)

                        with st.expander("Saturated vendors (Case B)", expanded=False):
                            df_sat = df_util[df_util["saturated"] == "YES"].copy().sort_values("vendor")
                            if df_sat.empty:
                                st.write("No saturated vendors.")
                            else:
                                df_show(df_sat)

                    # ---------------------------------------------------------
                    # 7) Final summary
                    # ---------------------------------------------------------
                    st.markdown("## Final comparison (Case A vs Case B)")

                    matched_A = len(resA.assignments) - len(resA.unmatched)
                    matched_B = len(resB.assignments) - len(resB.unmatched)

                    df_sum = pd.DataFrame([
                        {
                            "case": "A (Infinite items)",
                            "total_welfare": round(resA.total_welfare, 3),
                            "matched_players": matched_A,
                            "unmatched_players": len(resA.unmatched),
                            "runtime_ms": round(resA.runtime_ms, 2),
                        },
                        {
                            "case": "B (Limited capacity)",
                            "total_welfare": round(resB.total_welfare, 3),
                            "matched_players": matched_B,
                            "unmatched_players": len(resB.unmatched),
                            "runtime_ms": round(resB.runtime_ms, 2),
                        },
                    ])
                    df_show(df_sum)

                    if resA.total_welfare > 0:
                        drop = 100.0 * (resA.total_welfare - resB.total_welfare) / resA.total_welfare
                        st.caption(f"Welfare drop from Case A → Case B: {drop:.1f}% due to capacity constraints.")

                    # =========================================================
                    # OPTIONAL: Automatic sweeps
                    # =========================================================
                    st.markdown("## Additional Experiments — sweeps")

                    # ---------------------------------------------------------
                    # Capacity sweep: controlled capacity-only experiment
                    # ---------------------------------------------------------
                    with st.expander("Table 1 — Controlled capacity sweep", expanded=False):
                        cap_values = st.multiselect(
                            "capacity per vendor values",
                            options=[1, 2, 3, 4, 5],
                            default=[1, 2, 3, 4, 5],
                            help=(
                                "Controlled sweep: keeps S, players, vendor prices, vendor levels, "
                                "number of vendors, seed, and alpha fixed. Only changes the capacity "
                                "assigned to each vendor."
                            ),
                        )

                        run_cap_sweep = st.button("▶ Run controlled capacity sweep", key="run_cap_sweep")

                        if run_cap_sweep:
                            rows = []

                            # IMPORTANT:
                            # Use the current Task 3 players/vendors already generated by the main Task 3 run.
                            # Do NOT call run_task3_planner here, because that regenerates vendors.
                            base_players = players
                            base_vendors = vendors

                            for capM in cap_values:
                                capM = int(capM)

                                # Keep the same vendors, prices, and levels.
                                # Change only capacity.
                                vendors_cap = [
                                    Vendor(
                                        vid=v.vid,
                                        price=v.price,
                                        level=v.level,
                                        capacity=capM,
                                    )
                                    for v in base_vendors
                                ]

                                resA_cap = solve_infinite_items(
                                    base_players,
                                    vendors_cap,
                                    alpha=cfg.alpha,
                                )

                                resB_cap = solve_limited_items(
                                    base_players,
                                    vendors_cap,
                                    alpha=cfg.alpha,
                                )

                                A = float(resA_cap.total_welfare)
                                B = float(resB_cap.total_welfare)
                                drop = 100.0 * (A - B) / max(A, 1e-9)

                                rows.append({
                                    "capacity_per_vendor": capM,
                                    "total_capacity": capM * len(base_vendors),
                                    "welfare_A": round(A, 3),
                                    "welfare_B": round(B, 3),
                                    "welfare_drop_%": round(drop, 2),
                                    "unmatched_B": int(len(resB_cap.unmatched)),
                                    "matched_B": int(len(resB_cap.assignments) - len(resB_cap.unmatched)),
                                })

                            df_cap = pd.DataFrame(rows).sort_values("capacity_per_vendor")
                            st.session_state["task3_cap_sweep"] = df_cap

                        df_cap = st.session_state.get("task3_cap_sweep")
                        if df_cap is not None:
                            df_show(df_cap)

                            st.download_button(
                                "Download controlled capacity sweep (CSV)",
                                data=df_cap.to_csv(index=False).encode("utf-8"),
                                file_name="task3_controlled_capacity_sweep.csv",
                                mime="text/csv",
                                key="dl_cap_sweep",
                            )

                    # ---------------------------------------------------------
                    # Alpha sweep: controlled alpha-only experiment
                    # ---------------------------------------------------------
                    with st.expander("Table 2 — α sweep", expanded=False):
                        alpha_values = st.multiselect(
                            "α values",
                            options=[0.0, 0.25, 0.5, 0.7, 0.75, 1.0],
                            default=[0.0, 0.25, 0.5, 0.7, 0.75, 1.0],
                            help=(
                                "Controlled sweep: keeps S, players, vendor prices, vendor levels, "
                                "and capacities fixed. Only changes the trade-off between "
                                "affordability and security level."
                            ),
                        )

                        run_alpha_sweep = st.button("▶ Run α sweep", key="run_alpha_sweep")

                        if run_alpha_sweep:
                            rows = []

                            # Use current Task 3 players/vendors.
                            # This keeps budgets, vendor prices, vendor levels, and capacities fixed.
                            base_players = players
                            base_vendors = vendors

                            for a_val in alpha_values:
                                a_val = float(a_val)

                                resA_alpha = solve_infinite_items(
                                    base_players,
                                    base_vendors,
                                    alpha=a_val,
                                )

                                resB_alpha = solve_limited_items(
                                    base_players,
                                    base_vendors,
                                    alpha=a_val,
                                )

                                A = float(resA_alpha.total_welfare)
                                B = float(resB_alpha.total_welfare)
                                drop = 100.0 * (A - B) / max(A, 1e-9)

                                avg_A = A / max(len(base_players), 1)
                                avg_B = B / max(len(base_players), 1)

                                # Minimum compatible utility sanity check.
                                min_u = None
                                for p in base_players:
                                    for v in base_vendors:
                                        if v.price <= p.budget:
                                            security_score = v.level / 10.0
                                            affordability_score = 1.0 - (v.price / p.budget)
                                            u = (
                                                a_val * security_score
                                                + (1.0 - a_val) * affordability_score
                                            )
                                            if min_u is None or u < min_u:
                                                min_u = u

                                loads = resB_alpha.vendor_loads or {}
                                top2 = sorted(loads.items(), key=lambda kv: kv[1], reverse=True)[:2]
                                top2_str = ", ".join([f"{vid}({load})" for vid, load in top2]) if top2 else "-"

                                rows.append({
                                    "alpha": a_val,
                                    "welfare_A": round(A, 3),
                                    "welfare_B": round(B, 3),
                                    "welfare_drop_%": round(drop, 2),
                                    "avg_utility_A": round(avg_A, 4),
                                    "avg_utility_B": round(avg_B, 4),
                                    "unmatched_B": int(len(resB_alpha.unmatched)),
                                    "matched_B": int(len(resB_alpha.assignments) - len(resB_alpha.unmatched)),
                                    "min_compatible_utility": round(float(min_u), 4) if min_u is not None else None,
                                    "top_vendors_B": top2_str,
                                })

                            df_alpha = pd.DataFrame(rows).sort_values("alpha")
                            st.session_state["task3_alpha_sweep"] = df_alpha

                        df_alpha = st.session_state.get("task3_alpha_sweep")
                        if df_alpha is not None:
                            df_show(df_alpha)

                            st.download_button(
                                "Download α sweep (CSV)",
                                data=df_alpha.to_csv(index=False).encode("utf-8"),
                                file_name="task3_alpha_sweep.csv",
                                mime="text/csv",
                                key="dl_alpha_sweep",
                            )

    # -----------------------------------------------------------------
    # TAB 4: Task 4 — Secure Path Auction with VCG Payments
    # -----------------------------------------------------------------
    with tab4:
        st.subheader("Task 4 — Secure Path Auction with VCG Payments")

        # 1) choose S source
        S = None
        source = st.selectbox(
            "Choose S source",
            ["Task 2 (Shapley induced)", "Task 1 (strategic solution)"],
            index=0,
            key="task4_S_source",
        )

        if source.startswith("Task 2"):
            induced = st.session_state.get("task2_induced")
            if induced is not None and hasattr(induced, "S"):
                S = sorted(list(induced.S))
        else:
            task1 = st.session_state.get("task1_result")
            if task1 is not None and hasattr(task1, "final_actions"):
                S = sorted([i for i, a in task1.final_actions.items() if a == 1])

        if not S:
            st.warning(
                "No S found yet for the selected source.\n\n"
                "- If you chose Task 2: run Task 2 to compute Shapley-induced S.\n"
                "- If you chose Task 1: run Task 1 and ensure it stores a security set."
            )

            with st.expander("Debug: available session_state keys", expanded=False):
                st.write(sorted(list(st.session_state.keys())))

        else:
            st.write(f"Using |S| = {len(S)} : {S}")

            nodes = sorted(list(G.nodes()))
            if len(nodes) < 2:
                st.error("Graph must have at least 2 nodes.")
            else:
                st.markdown("### Auction settings")

                colA, colB, colC = st.columns(3)
                with colA:
                    s = st.selectbox("Source s", nodes, index=0, key="task4_s")
                with colB:
                    t_default = len(nodes) - 1 if len(nodes) > 1 else 0
                    t = st.selectbox("Target t", nodes, index=t_default, key="task4_t")
                with colC:
                    lam = st.slider(
                        "λ (penalty for unsecure nodes)",
                        0.0, 50.0, 5.0, 0.5,
                        key="task4_lam",
                    )

                t4_seed = st.number_input(
                    "Task 4 random seed",
                    min_value=0,
                    max_value=10_000,
                    value=42,
                    step=1,
                    key="task4_seed",
                )

                exclude_endpoints = st.checkbox(
                    "Exclude endpoints (do not pay s/t)",
                    value=True,
                    key="task4_excl",
                )

                st.info(
                    "Task 4 chooses an s–t path minimizing the reported path score: "
                    "sum of reported costs on the path plus λ times the number of unsecure nodes on the path. "
                    "VCG/Clarke pivot payments are then computed for the selected path nodes. "
                    "The misreport tools are empirical demonstrations; truthfulness follows from the VCG mechanism "
                    "under the stated model, not from a finite scan alone."
                )

                st.divider()
                st.markdown("### Optional truthfulness demo (one liar)")

                liar_on = st.checkbox(
                    "Simulate one lying agent",
                    value=False,
                    key="task4_liar_on",
                )

                liar_node = None
                liar_report = None

                # Defaults so later code never crashes
                liar_nodes_scan = []
                scan_on = False
                scan_min = scan_max = scan_step = None

                if liar_on:
                    liar_node = st.selectbox(
                        "Liar node",
                        nodes,
                        index=0,
                        key="task4_liar_node",
                    )

                    liar_report = st.slider(
                        "Liar reported price",
                        1, 200, 150, 1,
                        key="task4_liar_price",
                    )

                    st.markdown("#### Multi-misreport scan (optional)")

                    liar_nodes_scan = st.multiselect(
                        "Liar nodes to scan (can select multiple)",
                        options=nodes,
                        default=[liar_node] if liar_node is not None else [],
                        key="task4_liar_nodes_scan",
                    )

                    scan_on = st.checkbox(
                        "Run misreport scan (sweep reported prices for selected liar nodes)",
                        value=False,
                        key="task4_scan_on",
                    )

                    if scan_on:
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            scan_min = st.number_input(
                                "Scan min report",
                                1, 500, 1, 1,
                                key="task4_scan_min",
                            )
                        with c2:
                            scan_max = st.number_input(
                                "Scan max report",
                                1, 500, 200, 1,
                                key="task4_scan_max",
                            )
                        with c3:
                            scan_step = st.number_input(
                                "Scan step",
                                1, 200, 10, 1,
                                key="task4_scan_step",
                            )

                run_t4 = st.button("▶ Run Task 4 Auction", key="task4_run")

                if run_t4:
                    if s == t:
                        st.error("Source and target must be different.")
                        st.stop()

                    cfg = AuctionConfig(
                        lam=float(lam),
                        seed=int(t4_seed),
                        exclude_endpoints=bool(exclude_endpoints),
                    )

                    try:
                        # Always run truthful baseline
                        res_truth = run_task4_auction(
                            G, S, s, t, cfg,
                            liar_node=None,
                            liar_report=None,
                        )

                        st.session_state["task4_result_truthful"] = res_truth
                        st.session_state["task4_cfg"] = cfg

                        # If liar demo is ON, run liar scenario too
                        if liar_on and (liar_node is not None) and (liar_report is not None):
                            res_liar = run_task4_auction(
                                G, S, s, t, cfg,
                                liar_node=liar_node,
                                liar_report=liar_report,
                            )
                            st.session_state["task4_result_liar"] = res_liar
                        else:
                            st.session_state["task4_result_liar"] = None

                        # ------------------------------------------------------------
                        # Multi-misreport scan
                        # ------------------------------------------------------------
                        scan_summary = None
                        scan_details = {}

                        def _utility_of_node(res, node_id):
                            # Utility = payment - true cost if selected; otherwise 0.
                            if node_id not in res.winners:
                                return 0.0

                            pay = res.payments.get(node_id, 0.0)
                            if pay == float("inf"):
                                return float("inf")

                            return float(pay) - float(res.true_costs[node_id])

                        if (
                            liar_on
                            and scan_on
                            and liar_nodes_scan
                            and (scan_min is not None)
                            and (scan_max is not None)
                            and (scan_step is not None)
                        ):
                            scan_values = list(range(int(scan_min), int(scan_max) + 1, int(scan_step)))

                            # Safety guard to avoid UI freezing
                            MAX_RUNS = 400
                            total_runs = len(liar_nodes_scan) * len(scan_values)

                            if total_runs > MAX_RUNS:
                                st.warning(
                                    f"Scan too large ({total_runs} auction runs). "
                                    f"Reduce nodes/range or increase step. Max recommended is {MAX_RUNS}."
                                )
                            else:
                                truth_path = list(res_truth.path)

                                summary_rows = []
                                for ln in liar_nodes_scan:
                                    u_truth = _utility_of_node(res_truth, ln)

                                    rows = []
                                    best_u = -float("inf")
                                    best_r = None

                                    for r in scan_values:
                                        rr = run_task4_auction(
                                            G, S, s, t, cfg,
                                            liar_node=ln,
                                            liar_report=int(r),
                                        )

                                        u = _utility_of_node(rr, ln)

                                        if u > best_u:
                                            best_u = u
                                            best_r = int(r)

                                        rows.append({
                                            "liar_node": ln,
                                            "liar_report": int(r),
                                            "liar_selected": (ln in rr.winners),
                                            "liar_payment": rr.payments.get(ln, 0.0) if (ln in rr.winners) else 0.0,
                                            "liar_true_cost": rr.true_costs.get(ln, None),
                                            "liar_utility": u,
                                            "allocation_changed": (list(rr.path) != truth_path),
                                            "total_score": rr.total_score,
                                            "chosen_path": list(rr.path),
                                        })

                                    df_ln = pd.DataFrame(rows).sort_values("liar_utility", ascending=False)
                                    scan_details[ln] = df_ln

                                    summary_rows.append({
                                        "liar_node": ln,
                                        "truthful_utility": u_truth,
                                        "best_scan_utility": best_u,
                                        "best_report": best_r,
                                        "utility_increased?": (best_u > u_truth + 1e-9),
                                    })

                                scan_summary = pd.DataFrame(summary_rows).sort_values(
                                    "best_scan_utility",
                                    ascending=False,
                                )

                        st.session_state["task4_scan_summary"] = scan_summary
                        st.session_state["task4_scan_details"] = scan_details

                    except Exception as e:
                        st.error(f"Task 4 failed: {e}")

                res_truth = st.session_state.get("task4_result_truthful")
                res_liar = st.session_state.get("task4_result_liar")

                if res_truth is not None:
                    with st.expander("All node costs (true vs reported)", expanded=False):
                        if res_liar is None:
                            df_costs = pd.DataFrame([{
                                "node": n,
                                "secure": bool(res_truth.secure_mask.get(n, False)),
                                "true_cost": res_truth.true_costs.get(n),
                                "reported_cost": res_truth.reported_costs.get(n),
                                "lied?": (
                                    res_truth.true_costs.get(n)
                                    != res_truth.reported_costs.get(n)
                                ),
                            } for n in sorted(G.nodes())])

                            st.dataframe(df_costs, width="stretch")
                        else:
                            tabA, tabB = st.tabs(["Truthful costs", "Liar costs"])

                            with tabA:
                                dfA = pd.DataFrame([{
                                    "node": n,
                                    "secure": bool(res_truth.secure_mask.get(n, False)),
                                    "true_cost": res_truth.true_costs.get(n),
                                    "reported_cost": res_truth.reported_costs.get(n),
                                    "lied?": (
                                        res_truth.true_costs.get(n)
                                        != res_truth.reported_costs.get(n)
                                    ),
                                } for n in sorted(G.nodes())])

                                st.dataframe(dfA, width="stretch")

                            with tabB:
                                dfB = pd.DataFrame([{
                                    "node": n,
                                    "secure": bool(res_liar.secure_mask.get(n, False)),
                                    "true_cost": res_liar.true_costs.get(n),
                                    "reported_cost": res_liar.reported_costs.get(n),
                                    "lied?": (
                                        res_liar.true_costs.get(n)
                                        != res_liar.reported_costs.get(n)
                                    ),
                                } for n in sorted(G.nodes())])

                                st.dataframe(dfB, width="stretch")

                    st.divider()
                    st.markdown("## Result")

                    # ----------------------------
                    # Helper to render one result
                    # ----------------------------
                    def _render_result(res, title: str):
                        st.markdown(f"### {title}")

                        st.markdown("**Chosen path (allocation)**")
                        st.write(res.path)

                        st.markdown("**Path score breakdown**")

                        df_sum = pd.DataFrame([{
                            "s": res.s,
                            "t": res.t,
                            "λ": res.lam,
                            "reported_cost_sum": res.reported_cost_sum,
                            "unsecure_nodes_on_path": res.unsecure_count,
                            "total_score": res.total_score,
                            "runtime_ms": res.runtime_ms,
                        }])

                        st.dataframe(df_sum, width="stretch")

                        penalty = res.lam * res.unsecure_count
                        st.write(
                            f"Penalty part = λ × (#unsecure on path) = "
                            f"{res.lam} × {res.unsecure_count} = {penalty}"
                        )

                        st.markdown("**Winners and VCG payments**")

                        rows = []
                        for i in res.winners:
                            pay = res.payments.get(i, float("nan"))
                            alt_cost = res.alt_cost_without.get(i, float("nan"))

                            alt_path = None
                            if hasattr(res, "alt_path_without") and isinstance(res.alt_path_without, dict):
                                alt_path = res.alt_path_without.get(i)

                            rows.append({
                                "node": i,
                                "secure": bool(res.secure_mask.get(i, False)),
                                "true_cost": res.true_costs.get(i),
                                "reported_cost": res.reported_costs.get(i),
                                "node_weight_w(i)": res.node_weights.get(i),
                                "alt_cost_without_i": alt_cost,
                                "alt_path_without_i": str(alt_path) if alt_path is not None else "(no path)",
                                "payment_vcg": pay,
                                "profit(payment-true)": res.profits.get(i),
                                "critical": (pay == float("inf")),
                            })

                        df_pay = pd.DataFrame(rows)

                        if df_pay.empty:
                            st.info("No paid winners: the path uses only endpoints, or endpoints are excluded.")
                        else:
                            if "critical" not in df_pay.columns:
                                df_pay["critical"] = False
                            if "payment_vcg" not in df_pay.columns:
                                df_pay["payment_vcg"] = float("nan")

                            df_pay = df_pay.sort_values(
                                ["critical", "payment_vcg"],
                                ascending=[False, False],
                            )

                            st.dataframe(df_pay, width="stretch")

                        st.caption(
                            "Notes: total_score = sum(reported costs on path) + λ × (#unsecure nodes on path). "
                            "VCG payments pay each winning node its externality under the same path objective. "
                            "critical=True means removing that node disconnects s→t; in this model the alternative "
                            "cost is infinite, so the theoretical pivot payment is shown as ∞ rather than a normal finite payment."
                        )

                    # ----------------------------
                    # Show Truthful vs Liar
                    # ----------------------------
                    col1, col2 = st.columns(2)

                    with col1:
                        _render_result(res_truth, "Truthful baseline")

                    with col2:
                        if res_liar is None:
                            st.markdown("### Liar scenario")
                            st.info("Enable 'Simulate one lying agent' and rerun to see liar vs truthful.")
                        else:
                            _render_result(res_liar, "Liar scenario (one agent misreports)")

                    # ----------------------------
                    # One-line truthfulness statement
                    # ----------------------------
                    if res_liar is not None and liar_node is not None:
                        # Utility here = payment - true cost if selected; otherwise 0.
                        u_truth = (
                            float(res_truth.profits.get(liar_node, 0.0))
                            if liar_node in res_truth.winners
                            else 0.0
                        )

                        u_liar = (
                            float(res_liar.profits.get(liar_node, 0.0))
                            if liar_node in res_liar.winners
                            else 0.0
                        )

                        st.markdown("### Truthfulness check")
                        st.write(f"Liar node = {liar_node}")
                        st.write(f"Utility under truthful bid: **{u_truth:.4f}**")
                        st.write(f"Utility under lying bid: **{u_liar:.4f}**")

                        if u_liar <= u_truth + 1e-9:
                            st.success(
                                "In this tested misreport, the agent did not increase its utility. "
                                "This is consistent with VCG truthfulness."
                            )
                        else:
                            st.warning(
                                "Utility increased under this misreport. For a VCG mechanism this should not happen "
                                "under the same objective and consistent tie-breaking. Check payment computation, "
                                "endpoint exclusion, and whether utility is computed as payment minus true cost."
                            )

                    # ------------------------------------------------------------
                    # Multi-misreport scan results
                    # ------------------------------------------------------------
                    scan_summary = st.session_state.get("task4_scan_summary")
                    scan_details = st.session_state.get("task4_scan_details", {})

                    if scan_summary is not None:
                        st.divider()
                        st.markdown("## Multi-misreport scan")

                        st.markdown("### Summary: best tested misreport vs truthful report")
                        st.dataframe(scan_summary, width="stretch")

                        if "utility_increased?" in scan_summary.columns and scan_summary["utility_increased?"].any():
                            st.warning(
                                "Some tested misreports increased utility. For a VCG mechanism this should not happen "
                                "under the same objective and consistent tie-breaking. Check payment computation, "
                                "endpoint exclusion, and the utility comparison."
                            )
                        else:
                            st.success(
                                "In this finite scan, no tested misreport increased the agent’s utility. "
                                "This is empirical evidence consistent with VCG truthfulness, not a proof by itself."
                            )

                        with st.expander("Detailed scan tables per liar node", expanded=False):
                            for ln, df_ln in scan_details.items():
                                st.markdown(f"#### Liar node {ln}")
                                st.dataframe(df_ln, width="stretch")
                                st.caption(
                                    "Note: allocation_changed=True means the chosen path differs from the truthful path. "
                                    "The liar’s utility is computed as payment minus true cost if selected, and 0 if not selected."
                                )

    # =========================================================
    # REPORT TAB
    # =========================================================
    with tabR:
        st.subheader("Report / Export")
        st.write("This tab can later export tables/figures used in the LaTeX report.")
        st.caption("Keep all exports under outputs/ so your report can include them easily.")


if __name__ == "__main__":
    main()
