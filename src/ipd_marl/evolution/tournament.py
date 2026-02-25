"""Evolutionary tournament runner."""

from __future__ import annotations

import copy
import logging
import os

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from ipd_marl.envs.ipd_env import IPDEnv

from .agent_entry import AgentEntry
from .match import play_match
from .mutation import mutate
from .population import make_population

log = logging.getLogger(__name__)


class EvolutionaryTournament:
    """Manages the evolutionary tournament with a heterogeneous population.

    Parameters
    ----------
    cfg : DictConfig
        Full resolved Hydra config.
    run_dir : str
        Directory for saving artefacts.
    """

    def __init__(self, cfg: DictConfig, run_dir: str) -> None:
        self.cfg = cfg
        self.run_dir = run_dir

        evo = cfg.evolution
        self.generations: int = int(evo.generations)
        self.match_steps: int = int(evo.match_steps)
        self.mutation_noise: float = float(evo.mutation_noise)
        self.survival_rate: float = float(evo.survival_rate)
        self.num_opponents: int = int(evo.num_opponents)

        # Determine memory_length from the first learnable agent slot
        self._memory_length = 3  # fallback
        for slot in evo.population:
            if slot.name != "fixed_strategy":
                self._memory_length = int(slot.config.get("memory_length", 3))
                break

        self.env = IPDEnv(
            memory_length=self._memory_length,
            max_rounds=self.match_steps,
            noise=float(cfg.env.get("noise", 0.0)),
        )

        # Build population
        self.population: list[AgentEntry] = make_population(cfg)

    @property
    def pop_size(self) -> int:
        """Total number of agents in the population."""
        return len(self.population)

    def run(self) -> pd.DataFrame:
        """Run the full evolutionary tournament."""
        metrics: list[dict] = []

        for gen in range(1, self.generations + 1):
            fitness_scores = np.zeros(self.pop_size)

            # 1. Fitness evaluation — each agent plays against random opponents
            n_opp = min(self.num_opponents, self.pop_size - 1)

            for i, entry_i in enumerate(self.population):
                opp_indices = np.random.choice(
                    [j for j in range(self.pop_size) if j != i],
                    size=n_opp,
                    replace=False,
                )

                total_score = 0.0
                for opp_idx in opp_indices:
                    entry_j = self.population[opp_idx]
                    score_a, _ = play_match(
                        entry_i.agent,
                        entry_j.agent,
                        self.env,
                        self.match_steps,
                        train_a=entry_i.train,
                        train_b=entry_j.train,
                    )
                    total_score += score_a

                fitness_scores[i] = total_score / n_opp

            # 2. Collect metrics
            mean_fitness = float(np.mean(fitness_scores))
            max_fitness = float(np.max(fitness_scores))
            min_fitness = float(np.min(fitness_scores))
            std_fitness = float(np.std(fitness_scores))

            gen_metrics: dict = {
                "generation": gen,
                "mean_fitness": mean_fitness,
                "max_fitness": max_fitness,
                "min_fitness": min_fitness,
                "std_fitness": std_fitness,
            }

            # Per-type metrics
            agent_types = list(dict.fromkeys(e.name for e in self.population))
            for atype in agent_types:
                type_indices = [i for i, e in enumerate(self.population) if e.name == atype]
                type_scores = fitness_scores[type_indices]
                gen_metrics[f"mean_fitness_{atype}"] = float(np.mean(type_scores))

            metrics.append(gen_metrics)

            log.info(
                "Gen %d/%d | Mean: %.2f | Max: %.2f | %s",
                gen,
                self.generations,
                mean_fitness,
                max_fitness,
                " | ".join(f"{t}: {gen_metrics[f'mean_fitness_{t}']:.2f}" for t in agent_types),
            )

            # Separate fixed vs learnable
            fixed_entries = [e for e in self.population if e.is_fixed]
            learnable_indices = [i for i, e in enumerate(self.population) if not e.is_fixed]
            learnable_fitness = (
                fitness_scores[learnable_indices] if learnable_indices else np.array([])
            )

            # Fixed agents always survive (environmental pressure)
            new_population: list[AgentEntry] = list(fixed_entries)

            # Select top learnable agents
            if len(learnable_indices) > 0:
                sorted_learnable = np.argsort(learnable_fitness)[::-1]
                num_learnable_survivors = max(
                    1,
                    int(len(learnable_indices) * self.survival_rate),
                )
                survivor_learnable = sorted_learnable[:num_learnable_survivors]

                survivors = [self.population[learnable_indices[si]] for si in survivor_learnable]
                new_population.extend(survivors)

                # Fill remaining slots with mutated children
                target_size = self.pop_size
                while len(new_population) < target_size:
                    parent = survivors[np.random.randint(len(survivors))]
                    child_agent = copy.deepcopy(parent.agent)
                    mutate(child_agent, self.mutation_noise)
                    child_entry = AgentEntry(
                        agent=child_agent,
                        name=parent.name,
                        trainable=parent.trainable,
                        train=parent.train,
                    )
                    new_population.append(child_entry)

            self.population = new_population

        # Save best learnable agent
        learnable_agents = [e for e in self.population if not e.is_fixed]
        if learnable_agents:
            best = learnable_agents[0]
            ext = ".pt" if best.name == "dqn" else ".json"
            best_path = os.path.join(self.run_dir, f"best_agent_model{ext}")
            best.agent.save(best_path)
            log.info("Saved best agent (%s) to %s", best.name, best_path)

        return pd.DataFrame(metrics)
