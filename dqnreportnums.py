import os
import sys
import torch
import yaml

from types import SimpleNamespace

# Ensure the local package is importable when running this script directly
# (the `ipd_marl` package lives under the `src/` directory).
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from ipd_marl.envs.ipd_env import IPDEnv
from ipd_marl.agents.dqn import DQNAgent


MODEL_PATH = "agent_model.pt"
ENV_CONFIG_PATH = "resolved_config.yaml"
EXPERIMENTS_DIR = "experiments/agent_vs_titfortat_noise"

COOPERATION = 0
DEFECTION = 1

TEST_COOPERATION = tuple([COOPERATION] * 12)
TEST_COOPERATION_TENSOR = torch.tensor(TEST_COOPERATION, dtype=torch.float32).unsqueeze(0)
TEST_FORGIVENESS = tuple([COOPERATION] * 11 + [DEFECTION])
TEST_FORGIVENESS_TENSOR = torch.tensor(TEST_FORGIVENESS, dtype=torch.float32).unsqueeze(0)

output = []

for dir in os.listdir(EXPERIMENTS_DIR):
    if "dqn" not in dir:
        continue
    curr_model_path = os.path.join(EXPERIMENTS_DIR, dir, MODEL_PATH)
    curr_env_path = os.path.join(EXPERIMENTS_DIR, dir, ENV_CONFIG_PATH)
    curr_results = {
        "cooperation": 0,
        "forgive": 0,
        "noise": 0
    }
    with open(curr_env_path, "r") as envfile:
        # read yaml file
        yaml_file = yaml.safe_load(envfile)
        curr_results["noise"] = yaml_file["env"]["noise"]
        
        env = IPDEnv(yaml_file["agent"]["memory_length"], yaml_file["env"]["max_rounds"], noise=yaml_file["env"]["noise"])
        cfg = SimpleNamespace(
            lr=1e-3,
            gamma=yaml_file["agent"]["gamma"],
            epsilon=yaml_file["agent"]["epsilon"],
            batch_size=yaml_file["agent"]["batch_size"],
            buffer_capacity=10000,
            target_update_freq=100,
        )

        agent = DQNAgent(obs_dim=env.obs_dim, cfg=cfg)
        agent.load(curr_model_path)
        agent.policy_net.eval()
        with torch.no_grad():
            curr_results["cooperation"] = agent.act(TEST_COOPERATION_TENSOR.numpy()[0])
            curr_results["forgive"] = agent.act(TEST_FORGIVENESS_TENSOR.numpy()[0])

        output.append(curr_results)

for result in output:
    print(result)

