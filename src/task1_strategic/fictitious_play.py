from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Hashable, List, Optional
import random
import networkx as nx

from .game import GameParams, ActionProfile, is_pure_nash_equilibrium

Node = Hashable


@dataclass
class FPResult:
    converged: bool
    iterations: int
    final_actions: ActionProfile
    history_actions: List[ActionProfile]
    history_sizes: List[int]
    history_changed: List[Optional[int]]  # aligned with history_actions; history_changed[0] = None
    reached_pne: bool
    first_pne_iter: Optional[int]


def fictitious_play_run(
    G: nx.Graph,
    a0: ActionProfile,
    params: GameParams,
    T: int = 500,
    seed: int = 42,
    eps: float = 1e-12,
    alpha: float = 0.10,
    sequential: bool = True,
    shuffle_each_round: bool = True,
    store_history: bool = True,
    check_every: int = 1,
) -> FPResult:
    """
    Empirical Fictitious Play for the binary network-security game.

    This implementation is intentionally separated from BRD:

    - BRD responds to the current action profile.
    - Fictitious Play responds to empirical beliefs built from historical actions.
    - This function does NOT call best_response(G, i, a, params).
    - This function does NOT use epsilon-BRD or any current-profile correction.
    - PNE checking is used only as external validation/stopping.

    Belief model:
        p1[j] = empirical frequency with which player j has played action 1.

    Expected utilities for player i:
        EU_i(1) = -cost_i

        If player i plays 0, it is safe only if all neighbors play 1.
        Using the empirical-belief independence approximation:

            Pr(safe_i) = product_{j in N(i)} p1[j]

        Therefore:

            EU_i(0) = -penalty * (1 - Pr(safe_i))

    Tie-breaking:
        The player chooses action 1 only if EU_i(1) > EU_i(0) + eps.
        Otherwise, it chooses action 0. This favors free-riding when utilities are equal.

    Update modes:
        sequential=False:
            Synchronous FP. All players compute actions from the same belief vector,
            then all beliefs are updated together.

        sequential=True:
            True sequential FP. Players update one by one. After player i acts,
            the empirical belief p1[i] is updated immediately, so later players
            in the same round observe the updated belief for player i.
    Performance options:
        store_history=True:
            Store the full action profile at every iteration. This is useful for
            Streamlit visualization and per-iteration explanation.

        store_history=False:
            Store only the initial and final action profiles. This is recommended
            for CLI experiments on large graphs, because storing T full profiles
            can be memory-expensive.

        check_every:
            PNE validation frequency. check_every=1 checks after every iteration
            and gives exact stopping detection. Larger values reduce the cost of
            repeated global PNE checks, but the reported iteration becomes the
            first checked iteration where a PNE is detected.
    """

    # alpha is intentionally ignored here.
    # It is kept only for backward compatibility with older callers/UI.
    _ = alpha

    rnd = random.Random(seed)
    nodes: List[Node] = list(G.nodes())

    # Current pure action profile.
    a: ActionProfile = dict(a0)

    # History arrays aligned by time index.
    history_actions: List[ActionProfile] = [dict(a)]
    history_sizes: List[int] = [sum(1 for x in a.values() if x == 1)]
    history_changed: List[Optional[int]] = [None]

    # External validation: if initial profile is already PNE, stop.
    if is_pure_nash_equilibrium(G, a, params):
        return FPResult(
            converged=True,
            iterations=0,
            final_actions=dict(a),
            history_actions=history_actions,
            history_sizes=history_sizes,
            history_changed=history_changed,
            reached_pne=True,
            first_pne_iter=0,
        )

    # Empirical counts initialized with the initial observation a0.
    # obs_count[j] stores how many observations are available for player j.
    obs_count: Dict[Node, float] = {j: 1.0 for j in nodes}
    count1: Dict[Node, float] = {j: float(a[j]) for j in nodes}
    p1: Dict[Node, float] = {
        j: count1[j] / obs_count[j]
        for j in nodes
    }

    def expected_payoffs(i: Node) -> tuple[float, float]:
        """
        Return (EU_i(0), EU_i(1)) under current empirical beliefs.
        """
        # Utility of securing.
        u1 = -float(params.cost[i])

        # Probability that all neighbors of i are secure under empirical beliefs.
        prob_safe = 1.0
        for j in G.neighbors(i):
            pj = max(0.0, min(1.0, float(p1[j])))
            prob_safe *= pj

        # Utility of not securing under uncertainty.
        u0 = -float(params.penalty) * (1.0 - prob_safe)

        return u0, u1

    for t in range(1, T + 1):
        order = list(nodes)
        if shuffle_each_round:
            rnd.shuffle(order)

        changes = 0

        if sequential:
            # True sequential belief-based FP.
            # Each player updates using the current empirical beliefs.
            # After player i acts, p1[i] is updated immediately.
            for i in order:
                u0, u1 = expected_payoffs(i)
                chosen = 1 if (u1 > u0 + eps) else 0

                if chosen != a[i]:
                    changes += 1

                a[i] = chosen

                # Immediate observation update for player i only.
                obs_count[i] += 1.0
                count1[i] += float(chosen)
                p1[i] = count1[i] / obs_count[i]

        else:
            # Synchronous FP:
            # compute all actions from the same belief vector, then commit.
            new_a = dict(a)

            for i in order:
                u0, u1 = expected_payoffs(i)
                chosen = 1 if (u1 > u0 + eps) else 0

                if chosen != a[i]:
                    changes += 1

                new_a[i] = chosen

            a = new_a

            # Synchronous observation update for all players.
            for j in nodes:
                obs_count[j] += 1.0
                count1[j] += float(a[j])
                p1[j] = count1[j] / obs_count[j]

        # Log compact history always.
        # These scalar histories are cheap and useful even in CLI experiments.
        history_sizes.append(sum(1 for x in a.values() if x == 1))
        history_changed.append(changes)

        # Store full action profiles only when needed.
        # Streamlit uses these profiles for visualization and detailed logs.
        # Large CLI experiments should disable this to avoid storing T full
        # dictionaries of size |V|.
        if store_history:
            history_actions.append(dict(a))

        # External validation/stopping only.
        # check_every=1 gives exact detection. Larger values reduce the cost of
        # repeated global PNE checks on large graphs, but may detect convergence
        # later than the exact first PNE iteration.
        do_check = (check_every <= 1) or (t % check_every == 0) or (t == T)
        if do_check and is_pure_nash_equilibrium(G, a, params):
            # In summary-only mode, keep only initial and final profiles so that
            # downstream code still has a valid final action profile if needed.
            if not store_history:
                history_actions = [dict(a0), dict(a)]

            return FPResult(
                converged=True,
                iterations=t,
                final_actions=dict(a),
                history_actions=history_actions,
                history_sizes=history_sizes,
                history_changed=history_changed,
                reached_pne=True,
                first_pne_iter=t,
            )
    
    # In summary-only mode, avoid returning a long list of full profiles.
    # Keep only initial and final profiles for compatibility.
    if not store_history:
        history_actions = [dict(a0), dict(a)]

    return FPResult(
        converged=False,
        iterations=T,
        final_actions=dict(a),
        history_actions=history_actions,
        history_sizes=history_sizes,
        history_changed=history_changed,
        reached_pne=False,
        first_pne_iter=None,
    )