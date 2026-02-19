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


def compute_trust_margin(
    agent_rewards: Sequence[float], opponent_rewards: Sequence[float]
) -> float:
    """Trust margin ΔR = mean(agent_rewards) - mean(opponent_rewards).

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
        ``opp_coop_rate``, ``trust_margin``.

    Returns
    -------
    dict
        Keys: ``mean_reward``, ``mean_coop_rate``, ``mean_opp_coop_rate``,
        ``mean_trust_margin``.
    """
    return {
        "mean_reward": float(metrics_df["episode_reward"].mean()),
        "mean_coop_rate": float(metrics_df["coop_rate"].mean()),
        "mean_opp_coop_rate": float(metrics_df["opp_coop_rate"].mean()),
        "mean_trust_margin": float(metrics_df["trust_margin"].mean()),
    }
