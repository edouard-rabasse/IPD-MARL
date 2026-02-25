"""Fixed-strategy agent adapter wrapping an Axelrod strategy behind BaseAgent."""

from __future__ import annotations

import numpy as np

from ipd_marl.agents.base import BaseAgent
from ipd_marl.envs.axelrod_opponent import AxelrodOpponent


class FixedStrategyAgent(BaseAgent):
    """Deterministic agent that follows a named Axelrod strategy.

    This adapter wraps :class:`~ipd_marl.envs.axelrod_opponent.AxelrodOpponent`
    so that fixed strategies can participate in the same population as learnable
    agents (TabularQ, DQN) during evolutionary tournaments.

    Parameters
    ----------
    obs_dim : int
        Observation dimension (``2 * memory_length``).
    cfg : DictConfig
        Agent-level config containing ``strategy_name``.
    """

    def __init__(self, obs_dim: int, cfg) -> None:
        super().__init__(obs_dim=obs_dim, n_actions=2)
        self.strategy_name: str = str(cfg.strategy_name)
        self._opponent = AxelrodOpponent(strategy_name=self.strategy_name)

    # ---- BaseAgent interface ----
    def reset(self) -> None:
        """Re-initialise the underlying Axelrod strategy for a new episode."""
        self._opponent.reset()

    def act(self, obs: np.ndarray) -> int:
        """Return the strategy's next action (0=C, 1=D)."""
        return self._opponent.act()

    def observe(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Update the Axelrod strategy's history (no learning)."""
        # The action *we* took is what the Axelrod player played;
        # to reconstruct the opponent action, we look at next_obs.
        # However, in the evolution tournament the caller provides the
        # effective actions, so we can reconstruct from the last pair.
        # Since act() already returns the strategy's choice, we just need
        # to tell AxelrodOpponent what both sides played this round.
        #
        # Convention: `action` is OUR (fixed-strategy) action,
        # and the opponent's action is inferred from next_obs.
        # next_obs layout: pairs of (own_action, opp_action) for recent rounds.
        # The most recent pair is at the end.
        if len(next_obs) >= 2:
            opp_action = int(next_obs[-1])  # last element = opponent's action
            self._opponent.update(opp_action, action)

    # ---- persistence (no-ops) ----
    def save(self, path: str) -> None:
        """No-op: fixed strategies have no learned parameters."""

    def load(self, path: str) -> None:
        """No-op: fixed strategies have no learned parameters."""
