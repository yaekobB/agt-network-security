# src/task1_strategic/logging_utils.py
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from src.security.checks import is_security_set, is_minimal_security_set, coverage_stats
from src.task1_strategic.game import GameParams, ActionProfile, is_pure_nash_equilibrium


def actions_to_S(a: ActionProfile) -> set:
    return {i for i, ai in a.items() if ai == 1}


def changes_at(res: Any, t: int) -> Optional[int]:
    """
    Robustly read "changes at iteration t" across different result formats.

    We support two conventions:
      1) ALIGNED: history_changed has same length as history_actions, and t=0 is None
         => changes = history_changed[t]
      2) SHIFTED: history_changed has length len(history_actions)-1
         => changes = history_changed[t-1] for t>=1, and t=0 is None
    """
    if t == 0:
        return None

    hc = getattr(res, "history_changed", None)
    ha = getattr(res, "history_actions", None)
    if not hc or not ha:
        return None

    if len(hc) == len(ha):
        return hc[t]
    # shifted
    idx = t - 1
    if 0 <= idx < len(hc):
        return hc[idx]
    return None


def build_history_log(
    method: str,
    G: nx.Graph,
    params: GameParams,
    res: Any,
    include_minimal: bool = True,
) -> List[Dict[str, Any]]:
    """
    Build a unified per-iteration log for BRD/RM/FP.

    Each row:
      t, method, size_S, changed, pne, is_security_set, is_minimal_security_set,
      satisfied_nodes, secured_edges, fraction_satisfied_nodes, fraction_secured_edges,
      plus method-specific extras if present (e.g., RM gap).
    """
    ha: List[ActionProfile] = getattr(res, "history_actions", []) or []
    if not ha:
        # fallback: only final
        a_final = getattr(res, "final_actions")
        ha = [a_final]

    rows: List[Dict[str, Any]] = []
    for t, a_t in enumerate(ha):
        S = actions_to_S(a_t)
        pne = is_pure_nash_equilibrium(G, a_t, params)
        sec = is_security_set(G, S)
        minimal = is_minimal_security_set(G, S) if include_minimal else None
        cov = coverage_stats(G, S)

        row = {
            "t": t,
            "method": method,
            "size_S": len(S),
            "changed": changes_at(res, t),
            "pne": pne,
            "is_security_set": sec,
            "is_minimal_security_set": minimal,
            "satisfied_nodes": cov.satisfied_nodes,
            "secured_edges": cov.secured_edges,
            "fraction_satisfied_nodes": cov.fraction_satisfied_nodes,
            "fraction_secured_edges": cov.fraction_secured_edges,
        }

        # Optional RM / FP extras if present
        gap = getattr(res, "history_gap", None)
        if isinstance(gap, list) and t < len(gap):
            row["gap"] = gap[t]

        rows.append(row)

    return rows


def summarize_run(
    method: str,
    G: nx.Graph,
    params: GameParams,
    res: Any,
    runtime_ms: Optional[float] = None,
) -> Dict[str, Any]:
    """
    One-row summary for the comparison table.
    """
    a_final: ActionProfile = getattr(res, "final_actions")
    S_final = actions_to_S(a_final)

    summary = {
        "method": method,
        "reached_pne": is_pure_nash_equilibrium(G, a_final, params),
        "iterations": getattr(res, "iterations", None),
        "final_|S|": len(S_final),
        "is_security_set": is_security_set(G, S_final),
        "is_minimal_security_set": is_minimal_security_set(G, S_final),
    }

    cov = coverage_stats(G, S_final)
    summary["satisfied_nodes_%"] = round(100.0 * cov.fraction_satisfied_nodes, 2)
    summary["secured_edges_%"] = round(100.0 * cov.fraction_secured_edges, 2)

    if runtime_ms is not None:
        summary["runtime_ms"] = round(float(runtime_ms), 2)

    return summary
