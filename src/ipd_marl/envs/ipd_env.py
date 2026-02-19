"""Iterated Prisoner's Dilemma environment."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


# Standard Prisoner's Dilemma payoff matrix.
# Keys: (agent_action, opponent_action) → (reward_agent, reward_opponent)
# Actions: 0 = Cooperate, 1 = Defect
PAYOFF: dict[tuple[int, int], tuple[int, int]] = {
    (0, 0): (3, 3),  # Both cooperate
    (0, 1): (0, 5),  # Agent cooperates, opponent defects
    (1, 0): (5, 0),  # Agent defects, opponent cooperates
    (1, 1): (1, 1),  # Both defect
}


class IPDEnv:
    """Iterated Prisoner's Dilemma with configurable memory window and noise.

    Parameters
    ----------
    memory_length : int
        Number of past rounds visible to the agent (N). The observation is a
        flat vector of length ``2 * memory_length``.
    max_rounds : int
        Maximum number of rounds per episode.
    noise : float
        Probability σ that each player's action is *flipped* (independently)
        on any given round. 0.0 means deterministic.
    """

    def __init__(self, memory_length: int = 3, max_rounds: int = 100, noise: float = 0.0) -> None:
        self.memory_length = memory_length
        self.max_rounds = max_rounds
        self.noise = noise
        self.obs_dim: int = 2 * memory_length
        self._history: deque[tuple[int, int]] = deque(maxlen=memory_length)
        self._round: int = 0

    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        """Reset the environment and return the initial observation (all -1)."""
        self._history = deque(maxlen=self.memory_length)
        self._round = 0
        return self._get_obs()

    def step(
        self, agent_action: int, opponent_action: int
    ) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Advance one round.

        Parameters
        ----------
        agent_action : int
            Raw intended action of the agent (0=C, 1=D).
        opponent_action : int
            Raw intended action of the opponent (0=C, 1=D).

        Returns
        -------
        obs : np.ndarray
            Next observation.
        reward_agent : float
            Agent's payoff this round.
        done : bool
            Whether the episode is finished.
        info : dict
            Extra info (effective actions, round index, opponent reward).
        """
        eff_agent = self._maybe_flip(agent_action)
        eff_opp = self._maybe_flip(opponent_action)

        reward_agent, reward_opp = PAYOFF[(eff_agent, eff_opp)]

        self._history.append((eff_agent, eff_opp))
        self._round += 1

        done = self._round >= self.max_rounds
        obs = self._get_obs()

        info: dict[str, Any] = {
            "action_agent_effective": eff_agent,
            "action_opponent_effective": eff_opp,
            "round_index": self._round,
            "reward_opponent": reward_opp,
        }
        return obs, float(reward_agent), done, info

    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        """Build the flat observation from the recent history window.

        The vector has length ``2 * memory_length``.  Each pair of entries is
        ``(agent_action, opponent_action)`` for one past round.  Rounds that
        have not yet been played are padded with -1.
        """
        pad_len = self.memory_length - len(self._history)
        obs_list: list[int] = []
        for _ in range(pad_len):
            obs_list.extend([-1, -1])
        for a_act, o_act in self._history:
            obs_list.extend([a_act, o_act])
        return np.array(obs_list, dtype=np.int32)

    def _maybe_flip(self, action: int) -> int:
        """With probability ``self.noise``, flip the action."""
        if self.noise > 0.0 and np.random.random() < self.noise:
            return 1 - action
        return action
