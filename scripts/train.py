#!/usr/bin/env python
"""Train an IPD-MARL agent.

Usage examples::

    uv run python scripts/train.py experiment=baseline_tabular_vs_tft
    uv run python scripts/train.py agent=dqn opponent=axelrod_defector train.episodes=50 seed=123
    uv run python scripts/train.py opponent=self_play train.episodes=100
"""

from __future__ import annotations

import os

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from ipd_marl.training.evaluation import summarize_run
from ipd_marl.training.loops import train
from ipd_marl.utils.run_artifacts import make_run_dir, save_run_artifacts
from ipd_marl.utils.seed import set_seed


def _experiment_name(cfg: DictConfig) -> str:
    """Derive a human-readable experiment tag from the config."""
    opp = cfg.opponent.get("strategy_name", "self_play") or "self_play"
    opp_tag = str(opp).replace(" ", "").lower()
    return f"{cfg.agent.name}_vs_{opp_tag}"


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Hydra changes CWD to an outputs/ subfolder — go back to project root.
    orig_cwd: str = hydra.utils.get_original_cwd()

    # Collect Hydra overrides for metadata
    overrides: list[str] = list(HydraConfig.get().overrides.task)

    # Reproducibility
    set_seed(cfg.seed)

    # Create run directory under <project>/experiments/
    exp_name = _experiment_name(cfg)
    run_dir = make_run_dir(
        base_dir=os.path.join(orig_cwd, "experiments"),
        exp_name=exp_name,
    )
    save_run_artifacts(run_dir, cfg, overrides, cfg.seed)

    print(f">> Run directory: {run_dir}")
    print(f">> Agent: {cfg.agent.name}  |  Opponent: {cfg.opponent.type}")
    print(f">> Episodes: {cfg.train.episodes}  |  Rounds/ep: {cfg.env.max_rounds}")
    print()

    # Train
    metrics_df = train(cfg, run_dir)

    # Summary
    summary = summarize_run(metrics_df)
    print()
    print("=" * 60)
    print("RUN SUMMARY")
    print("=" * 60)
    print(f"  Run dir        : {run_dir}")
    print(f"  Mean reward    : {summary['mean_reward']:.2f}")
    print(f"  Mean coop rate : {summary['mean_coop_rate']:.2f}")
    print(f"  Mean opp coop  : {summary['mean_opp_coop_rate']:.2f}")
    print(f"  Mean ΔR        : {summary['mean_reward_difference']:.2f}")
    print("=" * 60)

    # Plotting
    try:
        from ipd_marl.utils.plotting import plot_run_metrics

        print(">> Generating plots...")
        plot_run_metrics(metrics_df, os.path.join(run_dir, "plots"))
        print(f"  Plots saved to : {os.path.join(run_dir, 'plots')}")
    except ImportError as e:
        print(f"[WARN] Could not generate plots: {e}")
    except Exception as e:
        print(f"[WARN] Error generating plots: {e}")


if __name__ == "__main__":
    main()
