"""Evolutionary tournament for heterogeneous IPD agent populations."""

from .agent_entry import AgentEntry
from .match import MatchResult, play_match
from .mutation import mutate
from .plotting import plot_evolution_metrics
from .population import make_population
from .tournament import EvolutionaryTournament

__all__ = [
    "AgentEntry",
    "EvolutionaryTournament",
    "MatchResult",
    "make_population",
    "mutate",
    "play_match",
    "plot_evolution_metrics",
]
