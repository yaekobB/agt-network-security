from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Hashable, List, Optional
import random
import networkx as nx

from .game import ActionProfile, is_pure_nash_equilibrium

Node = Hashable


@dataclass
class RMResult:
    converged: bool
    iterations: int
    final_actions: ActionProfile
    history_actions: List[ActionProfile]
    history_sizes: List[int]
    history_changed: List[Optional[int]]  # aligned: None at t=0
    history_gap: List[float]              # diagnostic: max positive cumulative regret


def regret_matching_run(
    G: nx.Graph,
    a0: ActionProfile,
    params,
    T: int = 500,
    seed: int = 42,
    tol: float = 1e-12,
    # ignored legacy args kept for backward compatibility with older callers/UI
    eps0: float = 0.40,
    eps_halflife: int = 200,
    sequential: bool = True,
    shuffle_each_round: bool = True,
    store_history: bool = True,
    check_every: int = 1,
) -> RMResult:
    """
    Pure Regret Matching for the binary network-security game.

    This implementation is intentionally separated from BRD:

    - BRD responds to the current action profile using best_response(...).
    - RM updates cumulative regrets from counterfactual utilities.
    - RM samples actions according to positive cumulative regret.
    - This function does NOT call best_response(...).
    - This function does NOT use epsilon-best-response.
    - PNE checking is used only as an external validation/stopping condition.

    Cumulative regret update:
        R_i(b) += u_i(b, a_-i) - u_i(a_i, a_-i)

    Action choice:
        Pr_i(b) ∝ max(R_i(b), 0)

    If all positive regrets are zero, the player randomizes uniformly.

    Important optimization:
        The old version computed u_i(0, a_-i) and u_i(1, a_-i) by creating
        two full action-profile copies for every player at every iteration.

        This version computes the same utilities locally from the payoff rule:

            u_i(1, a_-i) = -cost_i

            u_i(0, a_-i) = 0
                if all neighbors of i are secure,

            u_i(0, a_-i) = -penalty
                otherwise.

        This does NOT change the Regret Matching algorithm. It only avoids
        unnecessary O(n) dictionary copies inside the player loop.

    Note:
        Regret Matching is classically associated with convergence of average play
        toward correlated equilibrium. It is not guaranteed to reach a pure Nash
        equilibrium in every finite run. In this project, we report when the sampled
        pure profile reaches a PNE.

    Performance options:
        store_history=True:
            Store the full action profile at every iteration. This is useful for
            Streamlit visualization and detailed per-iteration logs.

        store_history=False:
            Store only the initial and final action profiles. This is recommended
            for large CLI experiments because RM may run for many iterations and
            full profile storage can become memory-expensive.

        check_every:
            PNE validation frequency. check_every=1 checks after every iteration.
            Larger values reduce repeated global PNE-check overhead, but the
            reported convergence iteration is the first checked iteration where
            a PNE is detected.
    """

    # Legacy parameters are intentionally ignored.
    # They remain only so older UI/compare calls do not break.
    _ = eps0
    _ = eps_halflife

    rnd = random.Random(seed)
    nodes: List[Node] = list(G.nodes())

    # Current pure action profile.
    a: ActionProfile = dict(a0)

    # History arrays. Keep the same structure as before for UI/CLI compatibility.
    history_actions: List[ActionProfile] = [dict(a)]
    history_sizes: List[int] = [sum(1 for x in a.values() if x == 1)]
    history_changed: List[Optional[int]] = [None]
    history_gap: List[float] = [0.0]

    # External validation: if the initial profile is already PNE, stop.
    if is_pure_nash_equilibrium(G, a, params):
        return RMResult(
            converged=True,
            iterations=0,
            final_actions=dict(a),
            history_actions=history_actions,
            history_sizes=history_sizes,
            history_changed=history_changed,
            history_gap=history_gap,
        )

    # Cumulative regrets per player and per action.
    R: Dict[Node, Dict[int, float]] = {i: {0: 0.0, 1: 0.0} for i in nodes}

    def action_probabilities(i: Node) -> Dict[int, float]:
        """
        Return the regret-matching mixed strategy for player i.
        """
        rp0 = max(R[i][0], 0.0)
        rp1 = max(R[i][1], 0.0)
        denom = rp0 + rp1

        if denom <= tol:
            return {0: 0.5, 1: 0.5}

        return {0: rp0 / denom, 1: rp1 / denom}

    def local_counterfactual_utilities(i: Node) -> tuple[float, float]:
        """
        Return (u_i(0, a_-i), u_i(1, a_-i)) without copying the full action profile.

        This is exactly equivalent to evaluating the Task 1 payoff for the two
        counterfactual actions of player i:

        - If i chooses 1, it pays its security cost:
              u_i(1, a_-i) = -c_i

        - If i chooses 0, it is safe only if all neighbors are secure:
              u_i(0, a_-i) = 0       if all j in N(i) have a_j = 1
              u_i(0, a_-i) = -P      otherwise

        We use a.get(j, 0) to match the safety convention used in game.py:
        missing actions are treated as non-secure.
        """
        u1 = -float(params.cost[i])

        safe_if_zero = True
        for j in G.neighbors(i):
            if a.get(j, 0) != 1:
                safe_if_zero = False
                break

        u0 = 0.0 if safe_if_zero else -float(params.penalty)
        return u0, u1

    for t in range(1, T + 1):
        order = list(nodes)
        if shuffle_each_round:
            rnd.shuffle(order)

        changed = 0
        max_pos_regret = 0.0

        if sequential:
            # Sequential RM variant.
            #
            # Still not BRD:
            # - utilities are used to update cumulative regrets,
            # - actions are sampled from positive regrets,
            # - no best_response(...) call is used.
            #
            # In sequential mode, a changes during the pass. The local utility helper
            # reads the current profile at the moment player i is updated, preserving
            # the same semantics as the old dict-copy implementation.
            for i in order:
                u0, u1 = local_counterfactual_utilities(i)
                ui = u1 if a[i] == 1 else u0

                R[i][0] += u0 - ui
                R[i][1] += u1 - ui

                max_pos_regret = max(
                    max_pos_regret,
                    max(R[i][0], 0.0),
                    max(R[i][1], 0.0),
                )

                probs = action_probabilities(i)
                chosen = 1 if rnd.random() < probs[1] else 0

                if chosen != a[i]:
                    changed += 1

                a[i] = chosen

        else:
            # Synchronous RM:
            # 1) update all regrets from the current pure profile,
            # 2) sample the next pure profile from regret-based mixed strategies.
            #
            # During the regret-update phase, a is fixed. We compute counterfactual
            # utilities locally without changing a.
            for i in order:
                u0, u1 = local_counterfactual_utilities(i)
                ui = u1 if a[i] == 1 else u0

                R[i][0] += u0 - ui
                R[i][1] += u1 - ui

                max_pos_regret = max(
                    max_pos_regret,
                    max(R[i][0], 0.0),
                    max(R[i][1], 0.0),
                )

            # Sample the next pure profile from the regret-based mixed strategies.
            new_a = dict(a)
            for i in order:
                probs = action_probabilities(i)
                new_a[i] = 1 if rnd.random() < probs[1] else 0

            changed = sum(1 for i in nodes if new_a[i] != a[i])
            a = new_a

        # Log compact histories always.
        # These are lightweight and useful for summaries/diagnostics.
        history_sizes.append(sum(1 for x in a.values() if x == 1))
        history_changed.append(changed)
        history_gap.append(max_pos_regret)

        # Store full action profiles only when needed.
        # Streamlit uses full profiles for visualization and detailed logging.
        # CLI runs on large graphs should disable this to avoid storing T full
        # dictionaries of size |V|.
        if store_history:
            history_actions.append(dict(a))

        # External validation/stopping only.
        # check_every=1 gives exact detection. Larger values reduce overhead from
        # repeated global PNE checks, but may detect convergence later.
        do_check = (check_every <= 1) or (t % check_every == 0) or (t == T)
        if do_check and is_pure_nash_equilibrium(G, a, params):
            # In summary-only mode, keep only initial and final profiles for
            # compatibility with downstream code.
            if not store_history:
                history_actions = [dict(a0), dict(a)]

            return RMResult(
                converged=True,
                iterations=t,
                final_actions=dict(a),
                history_actions=history_actions,
                history_sizes=history_sizes,
                history_changed=history_changed,
                history_gap=history_gap,
            )

    # In summary-only mode, avoid returning a long list of full profiles.
    # Keep only initial and final profiles for compatibility.
    if not store_history:
        history_actions = [dict(a0), dict(a)]

    return RMResult(
        converged=False,
        iterations=T,
        final_actions=dict(a),
        history_actions=history_actions,
        history_sizes=history_sizes,
        history_changed=history_changed,
        history_gap=history_gap,
    )