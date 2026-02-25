#!/usr/bin/env python
"""Evolutionary tournament for heterogeneous IPD agent populations.

Usage examples::

    uv run python scripts/evolution.py experiment=evolution
    uv run python scripts/evolution.py experiment=evolution evolution.generations=50 seed=7
"""

from __future__ import annotations

import os

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from ipd_marl.evolution import EvolutionaryTournament, plot_evolution_metrics
from ipd_marl.utils.run_artifacts import make_run_dir, save_run_artifacts
from ipd_marl.utils.seed import set_seed


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run the evolutionary tournament."""
    orig_cwd = hydra.utils.get_original_cwd()

    set_seed(cfg.seed)

    # Setup run directory
    exp_name = "evolution"
    run_dir = make_run_dir(
        base_dir=os.path.join(orig_cwd, "experiments"),
        exp_name=exp_name,
    )

    overrides = list(HydraConfig.get().overrides.task)
    save_run_artifacts(run_dir, cfg, overrides, cfg.seed)

    # Print summary
    pop_slots = cfg.evolution.population
    total_pop = sum(int(s.count) for s in pop_slots)
    print(f">> Run directory: {run_dir}")
    print(f">> Population: {total_pop} agents | Generations: {cfg.evolution.generations}")
    for slot in pop_slots:
        ckpt = slot.get("checkpoint", None)
        ckpt_str = f" (checkpoint: {ckpt})" if ckpt else ""
        print(f"  - {slot.count}x {slot.name} [train={slot.get('train', False)}]{ckpt_str}")
    print()

    # Run tournament
    tournament = EvolutionaryTournament(cfg, run_dir)
    metrics_df = tournament.run()

    # Save metrics
    metrics_path = os.path.join(run_dir, "evolution_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f">> Metrics saved to {metrics_path}")

    # Plotting
    try:
        plot_evolution_metrics(metrics_df, run_dir)
        print(f">> Plot saved to {os.path.join(run_dir, 'fitness_plot.png')}")
    except Exception as e:
        print(f"[WARN] Could not generate plot: {e}")


if __name__ == "__main__":
    main()
