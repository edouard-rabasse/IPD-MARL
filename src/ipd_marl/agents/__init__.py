"""Agent implementations for IPD-MARL."""

from __future__ import annotations

from ipd_marl.agents.base import BaseAgent
from ipd_marl.agents.tabular_q import TabularQAgent

__all__ = ["BaseAgent", "TabularQAgent", "DQNAgent", "FixedStrategyAgent", "make_agent"]


def make_agent(cfg) -> BaseAgent:
    """Instantiate an agent from a Hydra config node.

    DQNAgent and FixedStrategyAgent are imported lazily to avoid circular
    imports.
    """
    name = cfg.agent.name
    obs_dim = 2 * int(cfg.agent.memory_length)

    if name == "tabular_q":
        return TabularQAgent(obs_dim=obs_dim, cfg=cfg.agent)
    if name == "dqn":
        from ipd_marl.agents.dqn import DQNAgent  # lazy to break circular dep

        return DQNAgent(obs_dim=obs_dim, cfg=cfg.agent)
    if name == "fixed_strategy":
        from ipd_marl.agents.fixed_strategy import FixedStrategyAgent

        return FixedStrategyAgent(obs_dim=obs_dim, cfg=cfg.agent)
    raise ValueError(f"Unknown agent '{name}'. Available: ['tabular_q', 'dqn', 'fixed_strategy']")


def make_agent_from_slot(slot_cfg, memory_length: int = 3) -> BaseAgent:
    """Instantiate an agent from an evolution population slot config.

    This is used by the evolution tournament to create agents from the
    population roster, where each slot has its own ``name``, ``config``,
    and optional ``checkpoint`` path.

    Parameters
    ----------
    slot_cfg : DictConfig
        A single population slot with ``name``, ``config``, and optionally
        ``checkpoint``.
    memory_length : int
        Fallback memory length (used to compute ``obs_dim``).

    Returns
    -------
    BaseAgent
        The instantiated agent, optionally loaded from checkpoint.
    """
    name = slot_cfg.name
    agent_cfg = slot_cfg.config
    mem_len = int(agent_cfg.get("memory_length", memory_length))
    obs_dim = 2 * mem_len

    if name == "tabular_q":
        agent = TabularQAgent(obs_dim=obs_dim, cfg=agent_cfg)
    elif name == "dqn":
        from ipd_marl.agents.dqn import DQNAgent

        agent = DQNAgent(obs_dim=obs_dim, cfg=agent_cfg)
    elif name == "fixed_strategy":
        from ipd_marl.agents.fixed_strategy import FixedStrategyAgent

        agent = FixedStrategyAgent(obs_dim=obs_dim, cfg=agent_cfg)
    else:
        raise ValueError(
            f"Unknown agent '{name}'. Available: ['tabular_q', 'dqn', 'fixed_strategy']"
        )

    # Load checkpoint if specified
    checkpoint = slot_cfg.get("checkpoint", None)
    if checkpoint is not None:
        agent.load(str(checkpoint))

    return agent


# Lazy attribute so `from ipd_marl.agents import DQNAgent` still works.
def __getattr__(name: str):
    if name == "DQNAgent":
        from ipd_marl.agents.dqn import DQNAgent

        return DQNAgent
    if name == "FixedStrategyAgent":
        from ipd_marl.agents.fixed_strategy import FixedStrategyAgent

        return FixedStrategyAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
