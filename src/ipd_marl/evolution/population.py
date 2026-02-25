"""Population construction for evolutionary tournaments."""

from __future__ import annotations

import logging

from omegaconf import DictConfig

from ipd_marl.agents import make_agent_from_slot

from .agent_entry import AgentEntry

log = logging.getLogger(__name__)


def make_population(cfg: DictConfig) -> list[AgentEntry]:
    """Build the heterogeneous population from the evolution config.

    Parameters
    ----------
    cfg : DictConfig
        Full resolved Hydra config (must contain ``evolution.population``).

    Returns
    -------
    list[AgentEntry]
        One entry per agent instance in the population.
    """
    population: list[AgentEntry] = []

    for slot in cfg.evolution.population:
        count = int(slot.count)
        slot_type = str(slot.name)
        train = bool(slot.get("train", False))
        trainable = slot_type != "fixed_strategy"

        # Use the actual strategy name for fixed agents (e.g. "tit_for_tat")
        if slot_type == "fixed_strategy":
            display_name = str(slot.config.strategy_name).replace(" ", "_").lower()
        else:
            display_name = slot_type

        for i in range(count):
            agent = make_agent_from_slot(slot)
            entry = AgentEntry(
                agent=agent,
                name=display_name,
                trainable=trainable,
                train=train and trainable,
            )
            population.append(entry)
            log.debug(
                "Created agent %s (#%d/%d) train=%s checkpoint=%s",
                display_name,
                i + 1,
                count,
                entry.train,
                slot.get("checkpoint", None),
            )

    log.info(
        "Population created: %d agents (%s)",
        len(population),
        ", ".join(
            f"{sum(1 for e in population if e.name == n)}x {n}"
            for n in dict.fromkeys(e.name for e in population)
        ),
    )
    return population
