"""Abstract base class for all IPD agents."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseAgent(ABC):
    """Common interface every IPD agent must implement.

    Parameters
    ----------
    obs_dim : int
        Dimension of the observation vector (``2 * memory_length``).
    n_actions : int
        Number of discrete actions (always 2 for IPD: Cooperate / Defect).
    """

    def __init__(self, obs_dim: int, n_actions: int = 2) -> None:
        self.obs_dim = obs_dim
        self.n_actions = n_actions

    # ------ life-cycle hooks (concrete, overrideable) ------
    def reset(self) -> None:
        """Called at the start of every episode (no-op by default)."""

    def end_episode(self) -> None:
        """Called at the end of every episode (no-op by default)."""

    def update_epsilon(self) -> None:
        """Decay exploration rate (no-op by default for non-RL agents)."""

    # ------ core RL interface (abstract) ------
    @abstractmethod
    def act(self, obs: np.ndarray) -> int:
        """Select an action given the current observation.

        Returns
        -------
        int
            0 for Cooperate, 1 for Defect.
        """

    @abstractmethod
    def observe(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Process a transition (s, a, r, s', done)."""

    # ------ persistence (abstract) ------
    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the agent's learned parameters to *path*."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Restore the agent's learned parameters from *path*."""
