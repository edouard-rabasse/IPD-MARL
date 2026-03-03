"""Plotting utilities for IPD-MARL experiments."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ipd_marl.utils.plot_style import set_style


def plot_run_metrics(metrics_df: pd.DataFrame, output_dir: str) -> None:
    """Generate and save standard plots for a single training run.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame containing 'episode_reward', 'coop_rate', 'opp_coop_rate', etc.
    output_dir : str
        Directory to save the plots (e.g. 'experiments/.../plots').
    """
    set_style()
    os.makedirs(output_dir, exist_ok=True)

    # 1. Rolling Mean Reward
    plt.figure()
    window = max(1, len(metrics_df) // 10)
    rolling_reward = metrics_df["episode_reward"].rolling(window=window, min_periods=1).mean()
    sns.lineplot(data=metrics_df, x=metrics_df.index, y="episode_reward", alpha=0.3, label="Raw")
    sns.lineplot(
        x=metrics_df.index, y=rolling_reward, label=f"Rolling Mean (w={window})", color="C0"
    )
    plt.title("Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reward_curve.png"))
    plt.close()

    # 2. Cooperation Rates
    plt.figure()
    window = max(1, len(metrics_df) // 10)

    # Ensure columns exist
    if "coop_rate" in metrics_df.columns:
        sns.lineplot(
            x=metrics_df.index,
            y=metrics_df["coop_rate"].rolling(window=window, min_periods=1).mean(),
            label="Agent Coop Rate",
        )

    if "opp_coop_rate" in metrics_df.columns:
        sns.lineplot(
            x=metrics_df.index,
            y=metrics_df["opp_coop_rate"].rolling(window=window, min_periods=1).mean(),
            label="Opponent Coop Rate",
        )

    plt.title("Cooperation Rates")
    plt.xlabel("Episode")
    plt.ylabel("Cooperation Rate")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cooperation_rates.png"))
    plt.close()

    # 3. Reward difference / performance gap (ΔR)
    if "reward_difference" in metrics_df.columns:
        plt.figure()
        window = max(1, len(metrics_df) // 10)
        sns.lineplot(
            x=metrics_df.index,
            y=metrics_df["reward_difference"].rolling(window=window, min_periods=1).mean(),
            label="Reward Difference (Agent - Opponent)",
            color="purple",
        )
        plt.axhline(0, color="gray", linestyle="--", alpha=0.5)
        plt.title("Reward Difference (ΔR)")
        plt.xlabel("Episode")
        plt.ylabel("Reward Difference")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "reward_difference.png"))
        plt.close()
