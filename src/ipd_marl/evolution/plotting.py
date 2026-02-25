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
