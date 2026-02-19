#!/usr/bin/env python
"""Visualize and compare IPD-MARL experiment runs.

Usage::

    # Single run
    uv run python scripts/visualize.py runs=experiments/2026-02-18/123456_my_run

    # Comparison
    uv run python scripts/visualize.py \
        runs="experiments/run1,experiments/run2" output=comparison.png
"""

from __future__ import annotations

import os
import sys

import hydra
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig

from ipd_marl.utils.plot_style import set_style
from ipd_marl.utils.plotting import plot_run_metrics


def load_run_metrics(run_dir: str) -> pd.DataFrame:
    """Load metrics.csv from a run directory."""
    metrics_path = os.path.join(run_dir, "metrics.csv")
    if not os.path.isfile(metrics_path):
        raise FileNotFoundError(f"metrics.csv not found in {run_dir}")
    df = pd.read_csv(metrics_path)
    # Tag the dataframe with the run name for comparison
    df["run_name"] = os.path.basename(os.path.normpath(run_dir))
    return df


def plot_comparison(run_dirs: list[str], output_file: str) -> None:
    """Plot comparative metrics for multiple runs."""
    dfs = []
    for d in run_dirs:
        try:
            dfs.append(load_run_metrics(d))
        except FileNotFoundError as e:
            print(f"⚠ Skipping {d}: {e}", file=sys.stderr)

    if not dfs:
        print("No valid runs to plot.", file=sys.stderr)
        return

    combined_df = pd.concat(dfs, ignore_index=True)
    set_style()
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Plot Reward Comparison
    plt.figure()
    sns.lineplot(
        data=combined_df,
        x=combined_df.index
        // len(
            run_dirs
        ),  # Approximate episode index if concatenated (Need proper index alignment)
        y="episode_reward",
        hue="run_name",
    )
    # Better approach: reset index for each df before concat?
    # Let's re-load properly

    # Reload with proper index
    combined_df = pd.DataFrame()
    for d in run_dirs:
        try:
            df = load_run_metrics(d)
            df["episode"] = df.index
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        except Exception:
            pass

    # 1. Reward Comparison
    plt.figure()
    sns.lineplot(data=combined_df, x="episode", y="episode_reward", hue="run_name")
    plt.title("Reward Comparison")
    plt.savefig(output_file.replace(".png", "_reward.png"))
    plt.close()

    # 2. Cooperation Rate Comparison
    if "coop_rate" in combined_df.columns:
        plt.figure()
        sns.lineplot(data=combined_df, x="episode", y="coop_rate", hue="run_name")
        plt.title("Cooperation Rate Comparison")
        plt.ylim(-0.05, 1.05)
        plt.savefig(output_file.replace(".png", "_coop_rate.png"))
        plt.close()

    print(f"Saved comparison plots to {output_file.replace('.png', '_*.png')}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    orig_cwd = hydra.utils.get_original_cwd()

    if not hasattr(cfg, "runs") or cfg.runs is None:
        print("ERROR: please provide runs=<path1,path2,...>", file=sys.stderr)
        sys.exit(1)

    runs_arg = str(cfg.runs)
    run_paths = [
        os.path.join(orig_cwd, p.strip()) if not os.path.isabs(p) else p
        for p in runs_arg.split(",")
    ]

    # Single run mode
    if len(run_paths) == 1:
        print(f"Visualizing single run: {run_paths[0]}")
        try:
            df = load_run_metrics(run_paths[0])
            out_dir = os.path.join(run_paths[0], "plots")
            plot_run_metrics(df, out_dir)
            print(f"Plots saved to {out_dir}")
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    # Comparison mode
    else:
        print(f"Comparing {len(run_paths)} runs...")
        output_file = "comparison.png"
        if hasattr(cfg, "output") and cfg.output is not None:
            output_file = cfg.output

        if not os.path.isabs(output_file):
            output_file = os.path.join(orig_cwd, output_file)

        plot_comparison(run_paths, output_file)


if __name__ == "__main__":
    main()
