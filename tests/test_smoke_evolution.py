"""Smoke test: run a very short evolutionary tournament."""

from __future__ import annotations

import tempfile

from omegaconf import OmegaConf

from ipd_marl.utils.seed import set_seed


def _build_evolution_cfg():
    """Build a minimal evolution config (no Hydra global init required)."""
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
            "evolution": {
                "generations": 2,
                "match_steps": 5,
                "mutation_noise": 0.01,
                "survival_rate": 0.5,
                "num_opponents": 2,
                "population": [
                    {
                        "name": "tabular_q",
                        "count": 3,
                        "train": True,
                        "checkpoint": None,
                        "config": {
                            "lr": 0.1,
                            "gamma": 0.95,
                            "epsilon": 0.1,
                            "memory_length": 3,
                        },
                    },
                    {
                        "name": "fixed_strategy",
                        "count": 2,
                        "train": False,
                        "checkpoint": None,
                        "config": {
                            "strategy_name": "Tit For Tat",
                        },
                    },
                ],
            },
        }
    )


class TestSmokeEvolution:
    def test_short_tournament_runs(self):
        """A 2-generation tournament with mixed agents should not crash."""
        # Import here to avoid issues if evolution module has top-level side effects
        from ipd_marl.evolution import EvolutionaryTournament

        cfg = _build_evolution_cfg()
        set_seed(cfg.seed)

        with tempfile.TemporaryDirectory() as tmp:
            tournament = EvolutionaryTournament(cfg, tmp)
            df = tournament.run()

            assert len(df) == 2  # 2 generations
            assert "mean_fitness" in df.columns
            assert "max_fitness" in df.columns
            assert "mean_fitness_tabular_q" in df.columns
            assert "mean_fitness_tit_for_tat" in df.columns

    def test_population_sizes_match_config(self):
        """Population size should match sum of all slot counts."""
        from ipd_marl.evolution import make_population

        cfg = _build_evolution_cfg()
        population = make_population(cfg)

        assert len(population) == 5  # 3 + 2
        trainable = [e for e in population if e.train]
        fixed = [e for e in population if e.is_fixed]
        assert len(trainable) == 3
        assert len(fixed) == 2

    def test_dqn_in_population(self):
        """Tournament with DQN agents should also not crash."""
        from ipd_marl.evolution import EvolutionaryTournament

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
                "evolution": {
                    "generations": 2,
                    "match_steps": 5,
                    "mutation_noise": 0.01,
                    "survival_rate": 0.5,
                    "num_opponents": 1,
                    "population": [
                        {
                            "name": "dqn",
                            "count": 2,
                            "train": True,
                            "checkpoint": None,
                            "config": {
                                "lr": 0.001,
                                "gamma": 0.99,
                                "epsilon": 0.5,
                                "memory_length": 3,
                                "batch_size": 4,
                                "buffer_capacity": 100,
                                "target_update_freq": 10,
                            },
                        },
                        {
                            "name": "fixed_strategy",
                            "count": 1,
                            "train": False,
                            "checkpoint": None,
                            "config": {
                                "strategy_name": "Defector",
                            },
                        },
                    ],
                },
            }
        )
        set_seed(cfg.seed)

        with tempfile.TemporaryDirectory() as tmp:
            tournament = EvolutionaryTournament(cfg, tmp)
            df = tournament.run()
            assert len(df) == 2
