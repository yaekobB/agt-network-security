"""
src/task1_strategic/dynamics.py

Best Response Dynamics for Task 1.

Corrected version:
- BRD still updates players sequentially to their best response.
- After each full pass, we explicitly check whether the current profile is a PNE.
- This makes BRD iteration counting consistent with RM and FP:
  iterations = first pass after which the profile is a PNE.
- changed == 0 is still recorded as a useful diagnostic, but it is not the only stopping rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, List, Optional

import random
import networkx as nx

from src.task1_strategic.game import (
    ActionProfile,
    GameParams,
    best_response,
    is_pure_nash_equilibrium,
)

Node = Hashable


@dataclass
class BRDResult:
    converged: bool
    iterations: int
    final_actions: ActionProfile
    history_sizes: List[int]                  # |S| over time
    history_changed: List[Optional[int]]      # aligned with history_actions; history_changed[0] = None
    history_actions: List[ActionProfile]      # action profile at t=0,1,2,...

    # Extra compatibility/diagnostic fields
    reached_pne: bool
    first_pne_iter: Optional[int]


def init_actions(
    G: nx.Graph,
    mode: str = "zeros",
    p: float = 0.3,
    seed: int = 42,
) -> ActionProfile:
    """
    Initialize action profile.

    - zeros: all players choose 0
    - ones: all players choose 1
    - random: each player chooses 1 with probability p
    """
    nodes = list(G.nodes())

    if mode == "zeros":
        return {i: 0 for i in nodes}

    if mode == "ones":
        return {i: 1 for i in nodes}

    if mode == "random":
        rnd = random.Random(seed)
        return {i: (1 if rnd.random() < p else 0) for i in nodes}

    raise ValueError(f"Unknown init mode: {mode}")


def brd_run(
    G: nx.Graph,
    a0: ActionProfile,
    params: GameParams,
    max_iters: int = 500,
    shuffle_each_round: bool = True,
    seed: int = 42,
) -> BRDResult:
    """
    Run Best Response Dynamics.

    Important correction:
    The algorithm now checks PNE after each full pass and stops as soon as
    the current profile is a PNE. This makes BRD iteration counting comparable
    with RM and FP.

    Returns:
        converged:
            True if a PNE was reached within max_iters.
        iterations:
            First iteration/pass at which PNE is reached.
        final_actions:
            Final action profile.
        history_sizes:
            |S| over time, aligned with history_actions.
        history_changed:
            Number of action changes per pass, with None at t=0.
        history_actions:
            Action profiles over time.
        reached_pne:
            Same as converged, kept for consistency with FP/RM-style outputs.
        first_pne_iter:
            First pass where PNE was detected.
    """
    a: ActionProfile = dict(a0)
    nodes = list(G.nodes())
    rnd = random.Random(seed)

    history_actions: List[ActionProfile] = [dict(a)]
    history_sizes: List[int] = [sum(1 for v in a.values() if v == 1)]
    history_changed: List[Optional[int]] = [None]

    # Fair initial check: if a0 is already PNE, report 0 iterations.
    if is_pure_nash_equilibrium(G, a, params):
        return BRDResult(
            converged=True,
            iterations=0,
            final_actions=dict(a),
            history_sizes=history_sizes,
            history_changed=history_changed,
            history_actions=history_actions,
            reached_pne=True,
            first_pne_iter=0,
        )

    for it in range(1, max_iters + 1):
        order = list(nodes)
        if shuffle_each_round:
            rnd.shuffle(order)

        changed = 0

        # Sequential best-response pass.
        for i in order:
            br = best_response(G, i, a, params)
            if br != a[i]:
                a[i] = br
                changed += 1

        # Log profile after this pass.
        history_actions.append(dict(a))
        history_sizes.append(sum(1 for v in a.values() if v == 1))
        history_changed.append(changed)

        assert len(history_actions) == len(history_sizes)
        assert len(history_actions) == len(history_changed)

        # Correct/fair stopping condition:
        # stop at the first pass where the resulting profile is a PNE.
        if is_pure_nash_equilibrium(G, a, params):
            return BRDResult(
                converged=True,
                iterations=it,
                final_actions=dict(a),
                history_sizes=history_sizes,
                history_changed=history_changed,
                history_actions=history_actions,
                reached_pne=True,
                first_pne_iter=it,
            )

        # Safety fallback:
        # If no changes occurred but PNE check says False, something is inconsistent
        # between best_response() and is_pure_nash_equilibrium().
        if changed == 0:
            return BRDResult(
                converged=False,
                iterations=it,
                final_actions=dict(a),
                history_sizes=history_sizes,
                history_changed=history_changed,
                history_actions=history_actions,
                reached_pne=False,
                first_pne_iter=None,
            )

    return BRDResult(
        converged=False,
        iterations=max_iters,
        final_actions=dict(a),
        history_sizes=history_sizes,
        history_changed=history_changed,
        history_actions=history_actions,
        reached_pne=False,
        first_pne_iter=None,
    )