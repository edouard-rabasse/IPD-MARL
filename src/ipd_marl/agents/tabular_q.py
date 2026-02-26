"""Tabular Q-learning agent for the IPD."""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from ipd_marl.agents.base import BaseAgent


class TabularQAgent(BaseAgent):
    """ε-greedy tabular Q-learning agent.

    The state is the flattened observation vector (a tuple of ints) so that it
    can be used as a dictionary key.

    Parameters
    ----------
    obs_dim : int
        Observation dimension (``2 * memory_length``).
    cfg : DictConfig
        Agent-level Hydra config containing ``lr``, ``gamma``, ``epsilon``.
    """

    def __init__(self, obs_dim: int, cfg) -> None:
        super().__init__(obs_dim=obs_dim, n_actions=2)
        self.lr: float = float(cfg.lr)
        self.gamma: float = float(cfg.gamma)
        self.epsilon: float = float(cfg.epsilon)
        # Optimistic initialisation: start at max payoff (5.0) to drive exploration
        self.q_table: dict[tuple, np.ndarray] = defaultdict(
            lambda: np.full(self.n_actions, 5.0, dtype=np.float64)
        )

    # ---- helpers ----
    @staticmethod
    def _key(obs: np.ndarray) -> tuple:
        return tuple(obs.tolist())

    # ---- BaseAgent interface ----
    def act(self, obs: np.ndarray) -> int:
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.n_actions))
        q_values = self.q_table[self._key(obs)]
        max_q = np.max(q_values)
        ties = np.where(q_values == max_q)[0]
        return int(np.random.choice(ties))

    def update_epsilon(self, factor: float) -> None:
        """Decay exploration rate multiplicatively.

        Parameters
        ----------
        factor : float
            Multiplicative factor applied to ``epsilon`` (e.g. 0.995).
        """
        self.epsilon = max(0.0, self.epsilon * factor)

    def observe(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        key = self._key(obs)
        next_key = self._key(next_obs)
        best_next = float(np.max(self.q_table[next_key]))
        target = reward + self.gamma * best_next * (1.0 - float(done))
        self.q_table[key][action] += self.lr * (target - self.q_table[key][action])

    # ---- persistence ----
    def save(self, path: str) -> None:
        """Serialise Q-table to a JSON file."""
        serialisable = {str(k): v.tolist() for k, v in self.q_table.items()}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(serialisable, fh, indent=2)

    def load(self, path: str) -> None:
        """Load Q-table from a JSON file written by :meth:`save`."""
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        self.q_table = defaultdict(lambda: np.full(self.n_actions, 5.0, dtype=np.float64))
        for k_str, v in raw.items():
            # Key was str(tuple) → eval is safe because we wrote it ourselves
            key = tuple(json.loads(k_str.replace("(", "[").replace(")", "]")))
            self.q_table[key] = np.array(v, dtype=np.float64)
