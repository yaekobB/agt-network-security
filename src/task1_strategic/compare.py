# src/task1_strategic/compare.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import time
import networkx as nx

from src.task1_strategic.game import GameParams, ActionProfile
from src.task1_strategic.dynamics import brd_run
from src.task1_strategic.regret_matching import regret_matching_run
from src.task1_strategic.fictitious_play import fictitious_play_run

from src.task1_strategic.logging_utils import build_history_log, summarize_run


@dataclass
class CompareResult:
    summaries: list[dict]
    logs: dict[str, list[dict]]
    results: dict[str, Any]
    meta: Dict[str, Any] = field(default_factory=dict)


def run_compare_task1(
    G: nx.Graph,
    a0: ActionProfile,
    params: GameParams,
    *,
    seed: int = 42,
    brd_max_iters: int = 500,
    rm_T: int = 500,
    fp_T: int = 500,
    # pass-through knobs
    rm_kwargs: Optional[Dict[str, Any]] = None,
    fp_kwargs: Optional[Dict[str, Any]] = None,
    brd_kwargs: Optional[Dict[str, Any]] = None,
    # comparison/reporting controls
    build_logs: bool = True,
    meta: Optional[Dict[str, Any]] = None,
) -> CompareResult:
    """
    Run the three Task 1 dynamics on the same graph, initial action profile,
    and game parameters.

    Methods:
        - BRD: current-profile best response dynamics
        - RM: regret-matching learning dynamics
        - FP: empirical-belief fictitious play

    The function returns:
        summaries:
            one final summary row per method
        logs:
            per-iteration logs, if build_logs=True
        results:
            raw result objects
        meta:
            optional experiment metadata

    Notes:
        build_logs=True is useful for Streamlit visualization and convergence
        plots. For large CLI experiments, use build_logs=False to avoid
        expensive per-iteration validation/log construction.
    """
    rm_kwargs = rm_kwargs or {}
    fp_kwargs = fp_kwargs or {}
    brd_kwargs = brd_kwargs or {}

    results: Dict[str, Any] = {}
    summaries: list[dict] = []
    logs: dict[str, list[dict]] = {}

    # --- BRD ---
    t0 = time.perf_counter()
    res_brd = brd_run(
        G,
        dict(a0),
        params,
        max_iters=brd_max_iters,
        seed=seed,
        **brd_kwargs,
    )
    ms = (time.perf_counter() - t0) * 1000.0

    results["BRD"] = res_brd
    summaries.append(summarize_run("BRD", G, params, res_brd, runtime_ms=ms))
    if build_logs:
        logs["BRD"] = build_history_log("BRD", G, params, res_brd)

    # --- RM ---
    t0 = time.perf_counter()
    res_rm = regret_matching_run(
        G,
        dict(a0),
        params,
        T=rm_T,
        seed=seed,
        **rm_kwargs,
    )
    ms = (time.perf_counter() - t0) * 1000.0

    results["RM"] = res_rm
    summaries.append(summarize_run("RM", G, params, res_rm, runtime_ms=ms))
    if build_logs:
        logs["RM"] = build_history_log("RM", G, params, res_rm)

    # --- FP ---
    t0 = time.perf_counter()
    res_fp = fictitious_play_run(
        G,
        dict(a0),
        params,
        T=fp_T,
        seed=seed,
        **fp_kwargs,
    )
    ms = (time.perf_counter() - t0) * 1000.0

    results["FP"] = res_fp
    summaries.append(summarize_run("FP", G, params, res_fp, runtime_ms=ms))
    if build_logs:
        logs["FP"] = build_history_log("FP", G, params, res_fp)

    return CompareResult(
        summaries=summaries,
        logs=logs,
        results=results,
        meta=dict(meta or {}),
    )