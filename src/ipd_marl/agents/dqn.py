"""DQN agent for the IPD (PyTorch)."""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ipd_marl.agents.base import BaseAgent
from ipd_marl.training.replay_buffer import ReplayBuffer


class QNetwork(nn.Module):
    """Simple MLP: obs_dim → 64 → 64 → n_actions."""

    def __init__(self, obs_dim: int, n_actions: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x)


class DQNAgent(BaseAgent):
    """Deep Q-Network agent with experience replay and target network.

    Parameters
    ----------
    obs_dim : int
        Observation dimension (``2 * memory_length``).
    cfg : DictConfig
        Agent-level Hydra config containing ``lr``, ``gamma``, ``epsilon``,
        ``batch_size``, ``buffer_capacity``, ``target_update_freq``.
    """

    def __init__(self, obs_dim: int, cfg) -> None:
        super().__init__(obs_dim=obs_dim, n_actions=2)
        self.lr: float = float(cfg.lr)
        self.gamma: float = float(cfg.gamma)
        self.epsilon: float = float(cfg.epsilon)
        self.batch_size: int = int(cfg.batch_size)
        self.target_update_freq: int = int(cfg.target_update_freq)

        self.device = torch.device("cpu")
        self.policy_net = QNetwork(obs_dim, self.n_actions).to(self.device)
        self.target_net = copy.deepcopy(self.policy_net).to(self.device)
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.buffer = ReplayBuffer(capacity=int(cfg.buffer_capacity))
        self.step_count: int = 0

    # ---- helpers ----
    def _to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        return torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _sync_target(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    # ---- BaseAgent interface ----
    def act(self, obs: np.ndarray) -> int:
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.n_actions))
        with torch.no_grad():
            q_values = self.policy_net(self._to_tensor(obs))
        return int(q_values.argmax(dim=1).item())

    def observe(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.add(obs, action, reward, next_obs, done)
        if len(self.buffer) >= self.batch_size:
            self._train_step()
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self._sync_target()

    def _train_step(self) -> None:
        obs_b, act_b, rew_b, nobs_b, done_b = self.buffer.sample(self.batch_size)

        obs_t = torch.tensor(obs_b, dtype=torch.float32, device=self.device)
        act_t = torch.tensor(act_b, dtype=torch.long, device=self.device).unsqueeze(1)
        rew_t = torch.tensor(rew_b, dtype=torch.float32, device=self.device).unsqueeze(1)
        nobs_t = torch.tensor(nobs_b, dtype=torch.float32, device=self.device)
        done_t = torch.tensor(done_b, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.policy_net(obs_t).gather(1, act_t)
        with torch.no_grad():
            next_q = self.target_net(nobs_t).max(dim=1, keepdim=True).values
            target = rew_t + self.gamma * next_q * (1.0 - done_t)

        loss = nn.functional.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # ---- persistence ----
    def save(self, path: str) -> None:
        torch.save(
            {
                "policy_state_dict": self.policy_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "step_count": self.step_count,
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.step_count = checkpoint["step_count"]
