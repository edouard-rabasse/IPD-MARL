"""Smoke test: run a very short training loop and check artefacts."""

from __future__ import annotations

import os
import tempfile

from omegaconf import OmegaConf

from ipd_marl.training.loops import train
from ipd_marl.utils.run_artifacts import make_run_dir, save_run_artifacts
from ipd_marl.utils.seed import set_seed


def _build_smoke_cfg():
    """Build a minimal config dict (no Hydra global init required)."""
    return OmegaConf.create(
        {
            "seed": 0,
            "agent": {
                "name": "tabular_q",
                "lr": 0.1,
                "gamma": 0.95,
                "epsilon": 0.1,
                "memory_length": 3,
            },
            "env": {
                "max_rounds": 5,
                "noise": 0.0,
            },
            "opponent": {
                "type": "axelrod",
                "strategy_name": "Tit For Tat",
            },
            "train": {
                "episodes": 2,
                "log_interval": 1,
            },
            "eval": {
                "eval_episodes": 1,
            },
        }
    )


class TestSmokeTrain:
    def test_short_training_creates_artifacts(self):
        cfg = _build_smoke_cfg()
        set_seed(cfg.seed)

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = make_run_dir(base_dir=tmp, exp_name="smoke_test")
            save_run_artifacts(run_dir, cfg, overrides=["train.episodes=2"], seed=cfg.seed)

            df = train(cfg, run_dir)

            # Artefact checks
            assert os.path.isfile(os.path.join(run_dir, "metadata.json"))
            assert os.path.isfile(os.path.join(run_dir, "resolved_config.yaml"))
            assert os.path.isfile(os.path.join(run_dir, "metrics.csv"))

            # DataFrame shape
            assert len(df) == 2
            assert "episode_reward" in df.columns
            assert "coop_rate" in df.columns

    def test_dqn_smoke(self):
        """Ensure DQN agent also runs without crash (ultra-short)."""
        cfg = OmegaConf.create(
            {
                "seed": 1,
                "agent": {
                    "name": "dqn",
                    "lr": 0.001,
                    "gamma": 0.99,
                    "epsilon": 0.5,
                    "memory_length": 3,
                    "batch_size": 4,
                    "buffer_capacity": 100,
                    "target_update_freq": 10,
                },
                "env": {
                    "max_rounds": 5,
                    "noise": 0.0,
                },
                "opponent": {
                    "type": "axelrod",
                    "strategy_name": "Defector",
                },
                "train": {
                    "episodes": 2,
                    "log_interval": 1,
                },
                "eval": {
                    "eval_episodes": 1,
                },
            }
        )
        set_seed(cfg.seed)

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = make_run_dir(base_dir=tmp, exp_name="smoke_dqn")
            save_run_artifacts(run_dir, cfg, overrides=[], seed=cfg.seed)
            df = train(cfg, run_dir)
            assert len(df) == 2
            assert os.path.isfile(os.path.join(run_dir, "metrics.csv"))
