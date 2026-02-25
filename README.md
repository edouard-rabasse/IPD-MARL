# IPD-MARL

**Iterated Prisoner's Dilemma — Multi-Agent Reinforcement Learning**

Research project exploring cooperation, defection, and emergent strategies in
the repeated Prisoner's Dilemma through tabular Q-learning and Deep Q-Networks
(DQN).  Agents play against classic Axelrod tournament strategies (Tit For Tat,
Defector, Cooperator, Grudger …) or learn via self-play.

---

## Installation

Requires **Python ≥ 3.11** and [**uv**](https://docs.astral.sh/uv/).

```bash
uv sync          # creates .venv/ and installs all runtime + dev dependencies
```

To add packages, 
```bash
uv add <name of the package>     
```


## Quick start

```bash
# 1. Run the baseline experiment (Tabular Q vs Tit For Tat, 200 episodes)
uv run python scripts/train.py experiment=baseline_tabular_vs_tft

# 2. Custom run — DQN vs Defector, 50 episodes, 10 % noise, seed 123
uv run python scripts/train.py agent=dqn opponent=axelrod_defector \
    train.episodes=50 env.noise=0.1 seed=123

# 3. Self-play training
uv run python scripts/train.py opponent=self_play train.episodes=300

# 4. Evaluate a past run
uv run python scripts/evaluate.py +run_dir=experiments/2026-02-18/143022_tabular_q_vs_titfortat

# 5. Run evolutionary tournament (mixed population)
uv run python scripts/evolution.py experiment=evolution

# 6. Run tests
uv run pytest

# 7. Lint
uv run ruff check src/ scripts/ tests/

# 8. Compare runs
uv run python scripts/visualize.py \
    runs="experiments/run1,experiments/run2" output=comparison.png
```

## Experiment outputs

Every training run creates a **unique, timestamped directory** under
`experiments/`:

```
experiments/
└── 2026-02-18/
    └── 143022_tabular_q_vs_titfortat/
        ├── resolved_config.yaml   # full Hydra config (all defaults merged)
        ├── metadata.json          # timestamp, git hash, seed, package versions
        ├── metrics.csv            # per-episode: reward, coop_rate, opp_coop_rate, trust_margin
        └── agent_model.json       # agent checkpoint (.json for tabular, .pt for DQN)
```

To find *which config produced a run*, open `resolved_config.yaml`.  To find
*when and on which code version* it ran, open `metadata.json`.

## Hydra configuration

Configs live in `configs/` and are composable:

| Config group | Default         | Alternatives                             |
|-------------|-----------------|------------------------------------------|
| agent       | `tabular_q`     | `dqn`, `fixed_strategy`                  |
| env         | `ipd`           | —                                        |
| opponent    | `axelrod_titfortat` | `axelrod_defector`, `self_play`       |
| train       | `default`       | —                                        |
| eval        | `default`       | —                                        |
| experiment  | *(none)*        | `baseline_tabular_vs_tft`, `evolution`   |

Override any value from the CLI:

```bash
uv run python scripts/train.py agent.epsilon=0.2 train.episodes=1000 seed=7
```

Add new experiments by creating a YAML file in `configs/experiment/` — see
`PROJECT_BRAIN.md` for the recipe.

## Repository structure

```
├── configs/                # Hydra YAML configs (composable)
│   ├── config.yaml         # root config with defaults list
│   ├── agent/              # tabular_q.yaml, dqn.yaml
│   ├── env/                # ipd.yaml
│   ├── opponent/           # axelrod_titfortat.yaml, axelrod_defector.yaml, self_play.yaml
│   ├── train/              # default.yaml
│   ├── eval/               # default.yaml
│   └── experiment/         # baseline_tabular_vs_tft.yaml, evolution.yaml
├── src/ipd_marl/           # Python package (src layout)
│   ├── agents/             # BaseAgent, TabularQAgent, DQNAgent, FixedStrategyAgent
│   ├── envs/               # IPDEnv, AxelrodOpponent
│   ├── training/           # train loop, evaluation metrics, replay buffer
│   └── utils/              # seed, git info, run artefact helpers
├── scripts/
│   ├── train.py            # Hydra entry point for training
│   ├── evolution.py        # Evolutionary tournament with config-driven population
│   ├── evaluate.py         # Summarise a past run
│   └── visualize.py        # Compare runs
├── tests/
│   ├── test_agent_interface.py
│   ├── test_smoke_train.py
│   └── test_smoke_evolution.py
├── experiments/            # Timestamped run outputs (tracked in Git via .gitkeep)
├── pyproject.toml
├── ruff.toml
└── README.md               # ← you are here
```

## Key metrics

| Metric | Definition |
|--------|-----------|
| `episode_reward` | Sum of agent payoffs over one episode |
| `coop_rate` | Fraction of agent's effective actions = Cooperate |
| `opp_coop_rate` | Fraction of opponent's effective actions = Cooperate |
| `trust_margin` (ΔR) | `mean(agent_reward) − mean(opponent_reward)` |

## Payoff matrix

|  | Opponent C | Opponent D |
|--|-----------|-----------|
| **Agent C** | 3 , 3 | 0 , 5 |
| **Agent D** | 5 , 0 | 1 , 1 |

## License

MIT
