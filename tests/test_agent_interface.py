"""Tests for the agent interface contract (BaseAgent subclasses)."""

from __future__ import annotations

import os
import tempfile

import numpy as np
from omegaconf import OmegaConf

from ipd_marl.agents.dqn import DQNAgent
from ipd_marl.agents.tabular_q import TabularQAgent


# ---------- helpers ----------


def _make_tabular_cfg():
    return OmegaConf.create({"lr": 0.1, "gamma": 0.95, "epsilon": 0.1, "memory_length": 3})


def _make_dqn_cfg():
    return OmegaConf.create(
        {
            "lr": 0.001,
            "gamma": 0.99,
            "epsilon": 0.1,
            "memory_length": 3,
            "batch_size": 4,
            "buffer_capacity": 100,
            "target_update_freq": 10,
        }
    )


# ---------- TabularQAgent ----------


class TestTabularQAgent:
    def test_act_returns_valid_action(self):
        agent = TabularQAgent(obs_dim=6, cfg=_make_tabular_cfg())
        obs = np.array([-1, -1, -1, -1, -1, -1], dtype=np.int32)
        action = agent.act(obs)
        assert action in (0, 1)

    def test_observe_no_crash(self):
        agent = TabularQAgent(obs_dim=6, cfg=_make_tabular_cfg())
        obs = np.array([-1, -1, -1, -1, -1, -1], dtype=np.int32)
        next_obs = np.array([0, 1, -1, -1, -1, -1], dtype=np.int32)
        agent.observe(obs, 0, 3.0, next_obs, False)

    def test_save_load_roundtrip(self):
        agent = TabularQAgent(obs_dim=6, cfg=_make_tabular_cfg())
        obs = np.array([-1, -1, -1, -1, -1, -1], dtype=np.int32)
        next_obs = np.array([0, 1, -1, -1, -1, -1], dtype=np.int32)
        agent.observe(obs, 0, 3.0, next_obs, False)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "q_table.json")
            agent.save(path)
            assert os.path.isfile(path)

            agent2 = TabularQAgent(obs_dim=6, cfg=_make_tabular_cfg())
            agent2.load(path)
            action = agent2.act(obs)
            assert action in (0, 1)


# ---------- DQNAgent ----------


class TestDQNAgent:
    def test_act_returns_valid_action(self):
        agent = DQNAgent(obs_dim=6, cfg=_make_dqn_cfg())
        obs = np.array([-1, -1, -1, -1, -1, -1], dtype=np.int32)
        action = agent.act(obs)
        assert action in (0, 1)

    def test_observe_no_crash(self):
        agent = DQNAgent(obs_dim=6, cfg=_make_dqn_cfg())
        obs = np.array([-1, -1, -1, -1, -1, -1], dtype=np.int32)
        next_obs = np.array([0, 1, -1, -1, -1, -1], dtype=np.int32)
        agent.observe(obs, 0, 3.0, next_obs, False)

    def test_save_load_roundtrip(self):
        agent = DQNAgent(obs_dim=6, cfg=_make_dqn_cfg())
        obs = np.array([-1, -1, -1, -1, -1, -1], dtype=np.int32)
        agent.act(obs)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "dqn_model.pt")
            agent.save(path)
            assert os.path.isfile(path)

            agent2 = DQNAgent(obs_dim=6, cfg=_make_dqn_cfg())
            agent2.load(path)
            action = agent2.act(obs)
            assert action in (0, 1)
