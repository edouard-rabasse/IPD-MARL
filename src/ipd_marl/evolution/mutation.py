"""Mutation operators for evolutionary tournaments."""

from __future__ import annotations

import numpy as np

from ipd_marl.agents.base import BaseAgent


def mutate(agent: BaseAgent, noise: float) -> None:
    """Apply Gaussian noise mutation to an agent's learnable parameters.

    Parameters
    ----------
    agent : BaseAgent
        The agent to mutate.  Fixed-strategy agents are silently skipped.
    noise : float
        Standard deviation of the Gaussian noise.
    """
    # TabularQAgent — mutate Q-table values
    if hasattr(agent, "q_table"):
        for key in agent.q_table:
            perturbation = np.random.normal(0, noise, size=agent.q_table[key].shape)
            agent.q_table[key] += perturbation

    # DQNAgent — mutate network weights
    elif hasattr(agent, "policy_net"):
        import torch

        with torch.no_grad():
            for param in agent.policy_net.parameters():
                param.add_(torch.randn_like(param) * noise)
            # Sync target network after mutation
            if hasattr(agent, "_sync_target"):
                agent._sync_target()

    # FixedStrategyAgent — nothing to mutate (silently skip)
