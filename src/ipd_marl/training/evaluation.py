"""Evaluation metrics for IPD experiments."""

from __future__ import annotations

from typing import Sequence

import pandas as pd


def compute_coop_rate(actions: Sequence[int]) -> float:
    """Fraction of actions that are Cooperate (0).

    Parameters
    ----------
    actions : sequence of int
        Each element is 0 (Cooperate) or 1 (Defect).

    Returns
    -------
    float
        Cooperation rate in [0, 1].
    """
    if len(actions) == 0:
        return 0.0
    return float(sum(1 for a in actions if a == 0) / len(actions))


def compute_conditional_coop(
    actions_self: Sequence[int],
    actions_opp: Sequence[int],
) -> tuple[float, float]:
    """Conditional cooperation probabilities.

    Parameters
    ----------
    actions_self : sequence of int
        This player's actions per round (0=C, 1=D).
    actions_opp : sequence of int
        Opponent's actions per round (0=C, 1=D).

    Returns
    -------
    tuple[float, float]
        ``(p_c_given_opp_c, p_c_given_opp_d)`` — probability this player
        cooperates given the opponent cooperated (or defected) on the
        *previous* round.  Returns ``(0.0, 0.0)`` if fewer than 2 rounds.
    """
    if len(actions_self) < 2 or len(actions_opp) < 2:
        return 0.0, 0.0

    c_after_c = 0
    total_after_c = 0
    c_after_d = 0
    total_after_d = 0

    for t in range(1, len(actions_self)):
        if actions_opp[t - 1] == 0:  # opponent cooperated
            total_after_c += 1
            if actions_self[t] == 0:
                c_after_c += 1
        else:  # opponent defected
            total_after_d += 1
            if actions_self[t] == 0:
                c_after_d += 1

    p_c_given_c = float(c_after_c / total_after_c) if total_after_c > 0 else 0.0
    p_c_given_d = float(c_after_d / total_after_d) if total_after_d > 0 else 0.0
    return p_c_given_c, p_c_given_d


def compute_retaliation_rate(
    actions_self: Sequence[int],
    actions_opp: Sequence[int],
) -> float:
    """Retaliation rate: P(D_t | opp_D_{t-1}).

    Parameters
    ----------
    actions_self : sequence of int
        This player's actions (0=C, 1=D).
    actions_opp : sequence of int
        Opponent's actions (0=C, 1=D).

    Returns
    -------
    float
        Probability of defecting after the opponent defected.
    """
    if len(actions_self) < 2 or len(actions_opp) < 2:
        return 0.0

    d_after_opp_d = 0
    total_after_opp_d = 0

    for t in range(1, len(actions_self)):
        if actions_opp[t - 1] == 1:
            total_after_opp_d += 1
            if actions_self[t] == 1:
                d_after_opp_d += 1

    return float(d_after_opp_d / total_after_opp_d) if total_after_opp_d > 0 else 0.0


def compute_forgiveness_rate(
    actions_self: Sequence[int],
    actions_opp: Sequence[int],
) -> float:
    """Forgiveness rate: P(C_t | self_D_{t-1} & opp_D_{t-1}).

    Measures the probability of returning to cooperation after a round of
    mutual defection.

    Parameters
    ----------
    actions_self : sequence of int
        This player's actions (0=C, 1=D).
    actions_opp : sequence of int
        Opponent's actions (0=C, 1=D).

    Returns
    -------
    float
        Probability of cooperating after mutual defection.
    """
    if len(actions_self) < 2 or len(actions_opp) < 2:
        return 0.0

    c_after_dd = 0
    total_dd = 0

    for t in range(1, len(actions_self)):
        if actions_self[t - 1] == 1 and actions_opp[t - 1] == 1:
            total_dd += 1
            if actions_self[t] == 0:
                c_after_dd += 1

    return float(c_after_dd / total_dd) if total_dd > 0 else 0.0


def compute_reward_difference(
    agent_rewards: Sequence[float], opponent_rewards: Sequence[float]
) -> float:
    """Reward difference ΔR = mean(agent_rewards) - mean(opponent_rewards).

    Parameters
    ----------
    agent_rewards, opponent_rewards : sequence of float
        Per-round rewards for both players.

    Returns
    -------
    float
        Positive means agent earned more on average.
    """
    if len(agent_rewards) == 0:
        return 0.0
    mean_a = sum(agent_rewards) / len(agent_rewards)
    mean_o = sum(opponent_rewards) / len(opponent_rewards)
    return float(mean_a - mean_o)


def summarize_run(metrics_df: pd.DataFrame) -> dict[str, float]:
    """Aggregate per-episode metrics into a run-level summary.

    Parameters
    ----------
    metrics_df : DataFrame
        Must contain columns ``episode_reward``, ``coop_rate``,
        ``opp_coop_rate``, and ``reward_difference``.

    Returns
    -------
    dict
        Keys: ``mean_reward``, ``mean_coop_rate``, ``mean_opp_coop_rate``,
        ``mean_reward_difference``.
    """
    return {
        "mean_reward": float(metrics_df["episode_reward"].mean()),
        "mean_coop_rate": float(metrics_df["coop_rate"].mean()),
        "mean_opp_coop_rate": float(metrics_df["opp_coop_rate"].mean()),
        "mean_reward_difference": float(metrics_df["reward_difference"].mean()),
    }
