#!/usr/bin/env python
"""Summarise metrics from an existing experiment run.

Usage::

    uv run python scripts/evaluate.py run_dir=experiments/2026-02-18/143022_tabular_q_vs_titfortat
"""

from __future__ import annotations

import os
import sys

import hydra
import pandas as pd
from omegaconf import DictConfig

from ipd_marl.training.evaluation import summarize_run


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    orig_cwd = hydra.utils.get_original_cwd()

    if not hasattr(cfg, "run_dir") or cfg.run_dir is None:
        print("ERROR: please provide run_dir=<path> as a CLI override.", file=sys.stderr)
        sys.exit(1)

    run_dir = cfg.run_dir
    if not os.path.isabs(run_dir):
        run_dir = os.path.join(orig_cwd, run_dir)

    metrics_path = os.path.join(run_dir, "metrics.csv")
    if not os.path.isfile(metrics_path):
        print(f"ERROR: {metrics_path} not found.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(metrics_path)
    summary = summarize_run(df)

    print(f"Run: {run_dir}")
    print(f"  Episodes       : {len(df)}")
    print(f"  Mean reward    : {summary['mean_reward']:.2f}")
    print(f"  Mean coop rate : {summary['mean_coop_rate']:.2f}")
    print(f"  Mean opp coop  : {summary['mean_opp_coop_rate']:.2f}")
    print(f"  Mean ΔR        : {summary['mean_reward_difference']:.2f}")


if __name__ == "__main__":
    main()
