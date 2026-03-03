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
        # Learning-rate schedule: "robbins_monro" (default) or "fixed".
        # - "robbins_monro": alpha_t(s,a) = lr / N_t(s,a)
        # - "fixed": alpha_t(s,a) = lr
        self.lr_schedule: str = str(getattr(cfg, "lr_schedule", "robbins_monro"))
        # Exploration schedule: "glie" (default) or "fixed".
        # - "glie": harmonic decay epsilon_t = epsilon_0 / (1 + t)
        # - "fixed": keep epsilon constant at cfg.epsilon
        self.epsilon_schedule: str = str(getattr(cfg, "epsilon_schedule", "glie"))
        # ε initial value (used for both fixed and GLIE schedules).
        self.epsilon_init: float = float(cfg.epsilon)
        self.epsilon: float = self.epsilon_init
        self._eps_t: int = 0
        # Optimistic initialisation: start at max payoff (5.0) to drive exploration
        self.q_table: dict[tuple, np.ndarray] = defaultdict(
            lambda: np.full(self.n_actions, 5.0, dtype=np.float64)
        )
        # Visit-counts for per-(state, action) step sizes.
        # Used when lr_schedule="robbins_monro":
        #   alpha_t(s, a) = (lr / N_t(s, a))
        # which satisfies ∑_t alpha_t = ∞ and ∑_t alpha_t^2 < ∞.
        self.visit_counts: dict[tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=np.int64)
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

    def update_epsilon(self) -> None:
        """Update ε according to the chosen schedule."""
        if self.epsilon_schedule == "fixed":
            # Keep epsilon constant at its initial value.
            self.epsilon = self.epsilon_init
            return

        # Default: GLIE-style harmonic decay
        #   epsilon_t = epsilon_0 / (1 + t)
        # which ensures epsilon_t → 0 while ∑_t epsilon_t diverges.
        self._eps_t += 1
        self.epsilon = self.epsilon_init / (1.0 + float(self._eps_t))

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

        if self.lr_schedule == "fixed":
            alpha = self.lr
        else:
            # Default: Robbins–Monro per-(state, action) learning rate.
            self.visit_counts[key][action] += 1
            n_sa = float(self.visit_counts[key][action])
            alpha = self.lr / n_sa

        self.q_table[key][action] += alpha * (target - self.q_table[key][action])

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
