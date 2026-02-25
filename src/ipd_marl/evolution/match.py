"""Match play logic for evolutionary tournaments."""

from __future__ import annotations

from ipd_marl.agents.base import BaseAgent
from ipd_marl.envs.ipd_env import IPDEnv


def play_match(
    agent_a: BaseAgent,
    agent_b: BaseAgent,
    env: IPDEnv,
    steps: int,
    train_a: bool = False,
    train_b: bool = False,
) -> tuple[float, float]:
    """Play a single match between two agents.

    Parameters
    ----------
    agent_a, agent_b : BaseAgent
        The two competing agents.
    env : IPDEnv
        The environment instance.
    steps : int
        Number of rounds to play.
    train_a, train_b : bool
        If True, the corresponding agent calls ``observe()`` on each
        transition (i.e. it *learns* from the match in real time).

    Returns
    -------
    tuple[float, float]
        Total reward for agent_a and agent_b respectively.
    """
    obs_a = env.reset()
    obs_b = obs_a.copy()

    agent_a.reset()
    agent_b.reset()

    total_reward_a = 0.0
    total_reward_b = 0.0

    for _ in range(steps):
        action_a = agent_a.act(obs_a)
        action_b = agent_b.act(obs_b)

        next_obs_a, reward_a, done, info = env.step(action_a, action_b)

        # Build mirrored observation for agent B
        next_obs_b = next_obs_a.copy()
        obs_reshaped = next_obs_b.reshape(-1, 2)
        obs_reshaped = obs_reshaped[:, ::-1]
        next_obs_b = obs_reshaped.flatten()

        reward_b = info["reward_opponent"]

        # Optionally train each agent
        if train_a:
            agent_a.observe(obs_a, action_a, reward_a, next_obs_a, done)
        if train_b:
            agent_b.observe(obs_b, action_b, reward_b, next_obs_b, done)

        total_reward_a += reward_a
        total_reward_b += reward_b

        obs_a = next_obs_a
        obs_b = next_obs_b

        if done:
            break

    return total_reward_a, total_reward_b
