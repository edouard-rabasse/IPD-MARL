"""Main training loops for IPD agents."""

from __future__ import annotations

import os

import pandas as pd

from ipd_marl.agents import make_agent
from ipd_marl.envs.axelrod_opponent import AxelrodOpponent
from ipd_marl.envs.ipd_env import IPDEnv
from ipd_marl.training.evaluation import compute_coop_rate, compute_trust_margin


def _run_episode_vs_opponent(
    agent,
    opponent: AxelrodOpponent,
    env: IPDEnv,
) -> dict[str, float]:
    """Play one full episode of agent vs. fixed Axelrod opponent.

    Returns a dict with episode-level metrics.
    """
    obs = env.reset()
    opponent.reset()
    agent.reset()

    agent_actions: list[int] = []
    opp_actions: list[int] = []
    agent_rewards: list[float] = []
    opp_rewards: list[float] = []

    done = False
    while not done:
        a_action = agent.act(obs)
        o_action = opponent.act()

        next_obs, reward, done, info = env.step(a_action, o_action)

        # Record effective (post-noise) actions
        eff_a = info["action_agent_effective"]
        eff_o = info["action_opponent_effective"]

        agent.observe(obs, eff_a, reward, next_obs, done)
        opponent.update(eff_a, eff_o)

        agent_actions.append(eff_a)
        opp_actions.append(eff_o)
        agent_rewards.append(reward)
        opp_rewards.append(float(info["reward_opponent"]))

        obs = next_obs

    agent.end_episode()

    return {
        "episode_reward": sum(agent_rewards),
        "coop_rate": compute_coop_rate(agent_actions),
        "opp_coop_rate": compute_coop_rate(opp_actions),
        "trust_margin": compute_trust_margin(agent_rewards, opp_rewards),
    }


def _run_episode_self_play(
    agent_1,
    agent_2,
    env: IPDEnv,
) -> dict[str, float]:
    """Play one full episode of agent_1 vs. agent_2 (self-play)."""
    obs_1 = env.reset()
    # agent_2 sees the mirrored history — start with same initial obs
    obs_2 = obs_1.copy()
    agent_1.reset()
    agent_2.reset()

    a1_actions: list[int] = []
    a2_actions: list[int] = []
    a1_rewards: list[float] = []
    a2_rewards: list[float] = []

    done = False
    while not done:
        act_1 = agent_1.act(obs_1)
        act_2 = agent_2.act(obs_2)

        next_obs_1, reward_1, done, info = env.step(act_1, act_2)
        reward_2 = float(info["reward_opponent"])
        eff_1 = info["action_agent_effective"]
        eff_2 = info["action_opponent_effective"]

        # Build mirrored observation for agent_2 (swap columns)
        next_obs_2 = next_obs_1.copy()
        for i in range(0, len(next_obs_2), 2):
            next_obs_2[i], next_obs_2[i + 1] = next_obs_2[i + 1], next_obs_2[i]

        agent_1.observe(obs_1, eff_1, reward_1, next_obs_1, done)
        agent_2.observe(obs_2, eff_2, reward_2, next_obs_2, done)

        a1_actions.append(eff_1)
        a2_actions.append(eff_2)
        a1_rewards.append(reward_1)
        a2_rewards.append(reward_2)

        obs_1 = next_obs_1
        obs_2 = next_obs_2

    agent_1.end_episode()
    agent_2.end_episode()

    return {
        "episode_reward": sum(a1_rewards),
        "coop_rate": compute_coop_rate(a1_actions),
        "opp_coop_rate": compute_coop_rate(a2_actions),
        "trust_margin": compute_trust_margin(a1_rewards, a2_rewards),
    }


def train(cfg, run_dir: str) -> pd.DataFrame:
    """Run the full training loop and persist metrics.

    Parameters
    ----------
    cfg : DictConfig
        Resolved Hydra config.
    run_dir : str
        Directory where ``metrics.csv`` and agent checkpoints are saved.

    Returns
    -------
    pd.DataFrame
        Per-episode metrics.
    """
    env = IPDEnv(
        memory_length=int(cfg.agent.memory_length),
        max_rounds=int(cfg.env.max_rounds),
        noise=float(cfg.env.noise),
    )

    episodes: int = int(cfg.train.episodes)
    log_interval: int = int(cfg.train.log_interval)
    is_self_play: bool = cfg.opponent.type == "self_play"

    agent = make_agent(cfg)

    if is_self_play:
        agent_2 = make_agent(cfg)
    else:
        opponent = AxelrodOpponent(strategy_name=cfg.opponent.strategy_name)

    records: list[dict[str, float]] = []

    for ep in range(1, episodes + 1):
        if is_self_play:
            row = _run_episode_self_play(agent, agent_2, env)
        else:
            row = _run_episode_vs_opponent(agent, opponent, env)

        row["episode"] = ep
        records.append(row)

        if ep % log_interval == 0 or ep == episodes:
            print(
                f"[Episode {ep:>5}/{episodes}]  "
                f"reward={row['episode_reward']:.1f}  "
                f"coop={row['coop_rate']:.2f}  "
                f"opp_coop={row['opp_coop_rate']:.2f}  "
                f"ΔR={row['trust_margin']:.2f}"
            )

    # Persist metrics
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(run_dir, "metrics.csv"), index=False)

    # Save agent checkpoint
    ext = ".pt" if cfg.agent.name == "dqn" else ".json"
    agent.save(os.path.join(run_dir, f"agent_model{ext}"))

    return df
