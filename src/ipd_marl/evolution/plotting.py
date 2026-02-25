"""Plotting utilities for evolutionary tournament results."""

from __future__ import annotations

import logging
import os

import pandas as pd

log = logging.getLogger(__name__)


def plot_evolution_metrics(metrics_df: pd.DataFrame, run_dir: str) -> None:
    """Generate fitness evolution plots.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame from ``EvolutionaryTournament.run()``.
    run_dir : str
        Directory to save the plot.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    from ipd_marl.utils.plot_style import set_style

    set_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Overall fitness
    ax.plot(
        metrics_df["generation"],
        metrics_df["mean_fitness"],
        label="Mean (all)",
        linewidth=2,
        color="white",
        alpha=0.9,
    )
    ax.fill_between(
        metrics_df["generation"],
        metrics_df["mean_fitness"] - metrics_df["std_fitness"],
        metrics_df["mean_fitness"] + metrics_df["std_fitness"],
        alpha=0.2,
        color="white",
    )

    # Per-type fitness curves
    type_cols = [c for c in metrics_df.columns if c.startswith("mean_fitness_")]
    colors = sns.color_palette("tab20", len(type_cols))
    for idx, col in enumerate(type_cols):
        label = col.replace("mean_fitness_", "")
        color = colors[idx % len(colors)]
        ax.plot(
            metrics_df["generation"],
            metrics_df[col],
            label=label,
            linewidth=1.5,
            color=color,
            linestyle="--",
        )

    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (Avg Reward)")
    ax.set_title("Evolutionary Tournament Progress")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plot_path = os.path.join(run_dir, "fitness_plot.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Plot saved to %s", plot_path)

    # ---- Extended behavioral plots (if columns present) ----
    _plot_behavioral_metrics(metrics_df, run_dir)


def _plot_behavioral_metrics(metrics_df: pd.DataFrame, run_dir: str) -> None:
    """Generate a multi-panel behavioural-metrics figure.

    Produces ``evolution_behavioral.png`` with three subplots:
    1. Cooperation rate per agent type over generations.
    2. Conditional cooperation (P(C|C) vs P(C|D)) per type.
    3. Retaliation vs Forgiveness per type.

    Silently skipped if the required columns are absent.
    """
    if "mean_coop_rate" not in metrics_df.columns:
        return  # old-style CSV — nothing to plot

    import matplotlib.pyplot as plt
    import seaborn as sns

    from ipd_marl.utils.plot_style import set_style

    set_style()

    # Detect agent types from column names
    coop_cols = [c for c in metrics_df.columns if c.startswith("coop_rate_")]
    agent_types = [c.replace("coop_rate_", "") for c in coop_cols]

    if not agent_types:
        return

    colors = sns.color_palette("tab20", len(agent_types))
    gen = metrics_df["generation"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Panel 1: cooperation rate per type ---
    ax = axes[0]
    for idx, atype in enumerate(agent_types):
        col = f"coop_rate_{atype}"
        if col in metrics_df.columns:
            ax.plot(gen, metrics_df[col], label=atype, color=colors[idx])
    ax.set_xlabel("Generation")
    ax.set_ylabel("Cooperation Rate")
    ax.set_title("Cooperation Rate by Type")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: conditional cooperation ---
    ax = axes[1]
    for idx, atype in enumerate(agent_types):
        pc_c = f"p_c_given_c_{atype}"
        pc_d = f"p_c_given_d_{atype}"
        if pc_c in metrics_df.columns:
            ax.plot(gen, metrics_df[pc_c], color=colors[idx], linestyle="-",
                    label=f"{atype} P(C|C)")
        if pc_d in metrics_df.columns:
            ax.plot(gen, metrics_df[pc_d], color=colors[idx], linestyle="--",
                    label=f"{atype} P(C|D)")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Probability")
    ax.set_title("Conditional Cooperation")
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # --- Panel 3: retaliation & forgiveness ---
    ax = axes[2]
    for idx, atype in enumerate(agent_types):
        ret_col = f"retaliation_{atype}"
        forg_col = f"forgiveness_{atype}"
        if ret_col in metrics_df.columns:
            ax.plot(gen, metrics_df[ret_col], color=colors[idx], linestyle="-",
                    label=f"{atype} Retaliation")
        if forg_col in metrics_df.columns:
            ax.plot(gen, metrics_df[forg_col], color=colors[idx], linestyle=":",
                    label=f"{atype} Forgiveness")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Rate")
    ax.set_title("Retaliation & Forgiveness")
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(run_dir, "evolution_behavioral.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Behavioral plot saved to %s", path)
