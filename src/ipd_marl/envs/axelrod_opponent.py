"""Wrapper around the *axelrod* library to use Axelrod strategies as opponents."""

from __future__ import annotations

import axelrod as axl
from axelrod.action import Action

# Build a name → class mapping once at import time.
_STRATEGY_MAP: dict[str, type[axl.Player]] = {s.name: s for s in axl.all_strategies}


class AxelrodOpponent:
    """Stateful wrapper that lets an Axelrod strategy play round-by-round.

    Parameters
    ----------
    strategy_name : str
        Human-readable strategy name as listed in the axelrod library
        (e.g. ``"Tit For Tat"``, ``"Defector"``, ``"Cooperator"``).
    """

    def __init__(self, strategy_name: str) -> None:
        if strategy_name not in _STRATEGY_MAP:
            raise ValueError(
                f"Unknown Axelrod strategy '{strategy_name}'. "
                f"Available (first 10): {list(_STRATEGY_MAP)[:10]} ..."
            )
        self._strategy_cls = _STRATEGY_MAP[strategy_name]
        self._player: axl.Player = self._strategy_cls()
        # A dummy player object whose history represents the RL agent's moves
        # as seen by the Axelrod strategy.
        self._dummy: axl.Player = axl.Cooperator()
        self._first_round: bool = True

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Re-initialise internal state for a new episode."""
        self._player = self._strategy_cls()
        self._dummy = axl.Cooperator()
        self._first_round = True

    def act(self) -> int:
        """Return the next action (0=C, 1=D).

        Must be called *before* :meth:`update` on each round.
        """
        action: Action = self._player.strategy(self._dummy)
        return 0 if action == Action.C else 1

    def update(self, agent_action: int, opp_action: int) -> None:
        """Record both players' effective actions for this round.

        Parameters
        ----------
        agent_action : int
            The RL agent's effective action (0=C, 1=D).
        opp_action : int
            The opponent's own effective action (0=C, 1=D) — should be the
            value returned by :meth:`act` (or its noisy version).
        """
        a_axl = Action.C if agent_action == 0 else Action.D
        o_axl = Action.C if opp_action == 0 else Action.D
        # axelrod history: player stores (own_action, opponent_action).
        self._player.history.append(o_axl, a_axl)
        self._dummy.history.append(a_axl, o_axl)
        self._first_round = False

    # ------------------------------------------------------------------
    @classmethod
    def available_strategies(cls) -> list[str]:
        """Return all strategy names recognised by axelrod."""
        return sorted(_STRATEGY_MAP.keys())
