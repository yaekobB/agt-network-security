# src/task2_coalitional/experiments.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import networkx as nx

from .characteristic import CharParams, get_characteristic
from .shapley import approximate_shapley
from .induce_set import induce_minimal_security_set_from_ranking


@dataclass(frozen=True)
class CompareConfig:
    """
    Configuration for comparing v1/v2/v3 over multiple graph seeds.

    char_names:
        Characteristic functions to compare, usually ["v1", "v2", "v3"].

    K:
        Number of Monte Carlo permutations for Shapley approximation.

    alpha:
        Weight on node satisfaction versus edge coverage in v2/v3.

    gamma:
        Size-efficiency strength used by v3.

    prune:
        Whether to prune the Shapley-induced security set to enforce minimality.
    """
    char_names: List[str]
    K: int
    alpha: float
    gamma: float
    prune: bool = True


def _first_security_step_index(steps: list) -> Optional[int]:
    """
    Return the 1-based build step t where the coalition first becomes a security set.
    """
    for s in steps:
        if getattr(s, "is_security_set", False):
            return int(getattr(s, "t", 0))
    return None


def _build_size(induced) -> int:
    """
    Return the size at which the build phase first reached a security set.

    Prefer the explicit field from InducedSetResult. Fall back to scanning steps.
    """
    reached_size = getattr(induced, "reached_security_at_size", None)
    if reached_size is not None:
        return int(reached_size)

    first_t = _first_security_step_index(getattr(induced, "steps", []))
    if first_t is not None:
        return int(first_t)

    return len(getattr(induced, "steps", []))


def run_task2_characteristic_compare(
    graph_factory: Callable[[int], nx.Graph],
    seeds: Iterable[int],
    config: CompareConfig,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run Task 2 for each seed and each characteristic function.

    For every seed:
    - build one graph,
    - compute Shapley values for v1/v2/v3,
    - induce a security set from the Shapley ranking,
    - optionally prune to minimality,
    - report success/minimality/runtime/size.

    Batch-experiment optimization:
        Intermediate minimality checks during build/prune logging are disabled
        here because final security/minimality are still always checked by
        induce_minimal_security_set_from_ranking(...).
    """
    rows: List[Dict[str, Any]] = []
    meta = extra_meta or {}

    params = CharParams(
        alpha=float(config.alpha),
        gamma=float(config.gamma),
    )

    for seed in seeds:
        G = graph_factory(int(seed))

        for cname in config.char_names:
            v_fn = get_characteristic(cname)

            shap = approximate_shapley(
                G,
                v_fn,
                params,
                samples=int(config.K),
                seed=int(seed),
            )

            induced = induce_minimal_security_set_from_ranking(
                G,
                shap.ranking,
                prune=bool(config.prune),
                log_intermediate_minimality=False,
            )

            build_size = _build_size(induced)
            final_size = int(induced.size_S)

            rows.append({
                **meta,
                "seed": int(seed),
                "v": cname,
                "K": int(config.K),
                "alpha": float(config.alpha),
                "gamma": float(config.gamma),
                "runtime_ms": float(getattr(shap, "runtime_ms", 0.0)),
                "cache_size": int(getattr(shap, "cache_size", 0)),
                "cache_hits": int(getattr(shap, "cache_hits", 0)),
                "build_size": build_size,
                "final_size": final_size,
                "pruned_count": max(0, build_size - final_size),
                "final_security": bool(induced.final_is_security_set),
                "final_minimal": bool(induced.final_is_minimal),
            })

    df_runs = pd.DataFrame(rows)

    group_cols = [
        c for c in
        ["graph", "n", "k", "p", "m", "v", "K", "alpha", "gamma"]
        if c in df_runs.columns
    ]

    if not group_cols:
        group_cols = ["v"]

    df_summary = (
        df_runs
        .groupby(group_cols, dropna=False)
        .agg(
            runs=("seed", "count"),
            success_rate=("final_security", "mean"),
            minimal_rate=("final_minimal", "mean"),
            final_size_mean=("final_size", "mean"),
            final_size_std=("final_size", "std"),
            build_size_mean=("build_size", "mean"),
            build_size_std=("build_size", "std"),
            runtime_mean_ms=("runtime_ms", "mean"),
            runtime_std_ms=("runtime_ms", "std"),
        )
        .reset_index()
    )

    return df_runs, df_summary


def run_task2_preset_sweep(
    graph_factory: Callable[[int], nx.Graph],
    seeds: Iterable[int],
    presets: List[Dict[str, Any]],
    char_names: List[str],
    K: int,
    prune: bool = True,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run a fair preset sweep.

    For each seed:
    - build one graph G_s,
    - run every preset on the same graph,
    - for each preset, run all selected characteristic functions.

    Presets should be dictionaries such as:
        {"preset": "baseline", "alpha": 0.7, "gamma": 1.0}
    """
    rows: List[Dict[str, Any]] = []
    meta = extra_meta or {}

    for seed in seeds:
        G = graph_factory(int(seed))

        for preset in presets:
            preset_name = str(preset.get("preset", "unnamed"))
            alpha = float(preset.get("alpha", 0.7))
            gamma = float(preset.get("gamma", 1.0))

            params = CharParams(
                alpha=alpha,
                gamma=gamma,
            )

            for cname in char_names:
                v_fn = get_characteristic(cname)

                shap = approximate_shapley(
                    G,
                    v_fn,
                    params,
                    samples=int(K),
                    seed=int(seed),
                )

                induced = induce_minimal_security_set_from_ranking(
                    G,
                    shap.ranking,
                    prune=bool(prune),
                    log_intermediate_minimality=False,
                )

                build_size = _build_size(induced)
                final_size = int(induced.size_S)

                rows.append({
                    **meta,
                    "seed": int(seed),
                    "preset": preset_name,
                    "v": cname,
                    "K": int(K),
                    "alpha": alpha,
                    "gamma": gamma,
                    "runtime_ms": float(getattr(shap, "runtime_ms", 0.0)),
                    "cache_size": int(getattr(shap, "cache_size", 0)),
                    "cache_hits": int(getattr(shap, "cache_hits", 0)),
                    "build_size": build_size,
                    "final_size": final_size,
                    "pruned_count": max(0, build_size - final_size),
                    "final_security": bool(induced.final_is_security_set),
                    "final_minimal": bool(induced.final_is_minimal),
                })

    df_runs = pd.DataFrame(rows)

    group_cols = [
        c for c in
        ["graph", "n", "k", "p", "m", "preset", "v", "K", "alpha", "gamma"]
        if c in df_runs.columns
    ]

    if not group_cols:
        group_cols = ["preset", "v"]

    df_summary = (
        df_runs
        .groupby(group_cols, dropna=False)
        .agg(
            runs=("seed", "count"),
            success_rate=("final_security", "mean"),
            minimal_rate=("final_minimal", "mean"),
            final_size_mean=("final_size", "mean"),
            final_size_std=("final_size", "std"),
            build_size_mean=("build_size", "mean"),
            build_size_std=("build_size", "std"),
            runtime_mean_ms=("runtime_ms", "mean"),
            runtime_std_ms=("runtime_ms", "std"),
        )
        .reset_index()
    )

    return df_runs, df_summary