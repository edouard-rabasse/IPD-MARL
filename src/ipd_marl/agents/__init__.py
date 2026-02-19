"""Agent implementations for IPD-MARL."""

from __future__ import annotations

from ipd_marl.agents.base import BaseAgent
from ipd_marl.agents.tabular_q import TabularQAgent

__all__ = ["BaseAgent", "TabularQAgent", "DQNAgent", "make_agent"]


def make_agent(cfg) -> BaseAgent:
    """Instantiate an agent from a Hydra config node.

    DQNAgent is imported lazily to avoid a circular import between
    ``ipd_marl.agents`` and ``ipd_marl.training``.
    """
    name = cfg.agent.name
    obs_dim = 2 * int(cfg.agent.memory_length)

    if name == "tabular_q":
        return TabularQAgent(obs_dim=obs_dim, cfg=cfg.agent)
    if name == "dqn":
        from ipd_marl.agents.dqn import DQNAgent  # lazy to break circular dep

        return DQNAgent(obs_dim=obs_dim, cfg=cfg.agent)
    raise ValueError(f"Unknown agent '{name}'. Available: ['tabular_q', 'dqn']")


# Lazy attribute so `from ipd_marl.agents import DQNAgent` still works.
def __getattr__(name: str):
    if name == "DQNAgent":
        from ipd_marl.agents.dqn import DQNAgent

        return DQNAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
