"""Agent entry dataclass for evolution populations."""

from __future__ import annotations

from dataclasses import dataclass

from ipd_marl.agents.base import BaseAgent


@dataclass
class AgentEntry:
    """Metadata for one agent in the population."""

    agent: BaseAgent
    name: str
    trainable: bool
    train: bool  # whether this agent actually trains during matches

    @property
    def is_fixed(self) -> bool:
        """Return True if this is a fixed-strategy (non-trainable) agent."""
        return not self.trainable
