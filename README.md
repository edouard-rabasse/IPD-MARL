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

# 6. Run the full experiment grid (10 conditions x 3 seeds = 30 runs)
uv run python scripts/run_experiments.py

# 7. Analyze results and generate report
uv run python scripts/analyze_experiments.py
# -> experiments/analysis.md  (report)
# -> experiments/figures/     (11 plots)

# 8. Run tests
uv run pytest

# 9. Lint
uv run ruff check src/ scripts/ tests/

# 10. Compare runs
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
| env         | `ipd`           | ---                                        |
| opponent    | `axelrod_titfortat` | `axelrod_defector`, `axelrod_cooperator`, `self_play` |
| train       | `default`       | ---                                        |
| eval        | `default`       | ---                                        |
| experiment  | *(none)*        | see table below                            |

### Experiment configs

| Experiment | Description |
|-----------|-------------|
| `baseline_tabular_vs_tft` | Tabular Q vs Tit For Tat |
| `baseline_tabular_vs_defector` | Tabular Q vs Defector |
| `baseline_tabular_vs_cooperator` | Tabular Q vs Cooperator |
| `dqn_vs_cooperator` | DQN vs Cooperator |
| `dqn_vs_defector` | DQN vs Defector |
| `dqn_vs_tft` | DQN vs Tit For Tat |
| `evolution` | Default evolution (3 Tabular + 3 DQN) |
| `evolution_6t_0d` | Evolution: 6 Tabular, 0 DQN |
| `evolution_4t_2d` | Evolution: 4 Tabular, 2 DQN |
| `evolution_3t_3d` | Evolution: 3 Tabular, 3 DQN |
| `evolution_2t_4d` | Evolution: 2 Tabular, 4 DQN |
| `evolution_0t_6d` | Evolution: 0 Tabular, 6 DQN |
| `evolution_6t_0d_axelrod` | 6T/0D + TFT, Defector, Cooperator, WSLS |
| `evolution_4t_2d_axelrod` | 4T/2D + Axelrod agents |
| `evolution_3t_3d_axelrod` | 3T/3D + Axelrod agents |
| `evolution_2t_4d_axelrod` | 2T/4D + Axelrod agents |
| `evolution_0t_6d_axelrod` | 0T/6D + Axelrod agents |

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
│   ├── run_experiments.py  # Run the full 10-condition x 3-seed experiment grid
│   ├── analyze_experiments.py # Aggregate results, generate plots, write report
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
| `trust_margin` (dR) | `mean(agent_reward) - mean(opponent_reward)` |
| `p_c_given_c` | P(Cooperate \| opponent cooperated last round) |
| `p_c_given_d` | P(Cooperate \| opponent defected last round) |
| `retaliation` | P(Defect \| opponent defected last round) |
| `forgiveness` | P(Cooperate \| mutual defection last round) |

## Payoff matrix

|  | Opponent C | Opponent D |
|--|-----------|-----------|
| **Agent C** | 3 , 3 | 0 , 5 |
| **Agent D** | 5 , 0 | 1 , 1 |

## License

MIT
