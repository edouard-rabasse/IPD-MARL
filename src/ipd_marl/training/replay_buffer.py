"""Experience replay buffer used by the DQN agent."""

from __future__ import annotations

from collections import deque

import numpy as np


class ReplayBuffer:
    """Fixed-capacity FIFO replay buffer storing (s, a, r, s', done) tuples.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to store.
    """

    def __init__(self, capacity: int = 10_000) -> None:
        self._buffer: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=capacity
        )

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Store a single transition."""
        self._buffer.append((obs, action, reward, next_obs, done))

    def sample(
        self, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a random mini-batch of transitions.

        Returns
        -------
        obs, actions, rewards, next_obs, dones
            Each as a NumPy array with ``batch_size`` rows.
        """
        indices = np.random.choice(len(self._buffer), size=batch_size, replace=False)
        batch = [self._buffer[i] for i in indices]
        obs_arr = np.array([t[0] for t in batch], dtype=np.float32)
        act_arr = np.array([t[1] for t in batch], dtype=np.int64)
        rew_arr = np.array([t[2] for t in batch], dtype=np.float32)
        nobs_arr = np.array([t[3] for t in batch], dtype=np.float32)
        done_arr = np.array([t[4] for t in batch], dtype=np.float32)
        return obs_arr, act_arr, rew_arr, nobs_arr, done_arr

    def __len__(self) -> int:
        return len(self._buffer)
