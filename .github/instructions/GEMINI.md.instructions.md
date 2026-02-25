# PROJECT_BRAIN.md — IPD-MARL Charter

> This file is the **single source of truth** for rules that any developer
> (human or AI agent) must follow when modifying this repository.

---

## 1. README Rule

**If you modify code, configs, scripts, or the directory structure, you MUST
update `README.md` in the same commit.**

No exception.  The README must always reflect:

- Available CLI commands and their syntax.
- Config group options in the Hydra table.
- Repository structure listing.
- Any new dependency or setup step.

---

## 2. Experiment Separation

### 2.1 One folder per run — never overwrite

Every invocation of `scripts/train.py` creates a **new** directory:

```
experiments/YYYY-MM-DD/HHMMSS_<experiment_tag>/
```

You must **never** write results into an existing run folder.  If you need to
re-run, let the system create a new timestamped directory.

### 2.2 Required artefacts per run

Each run directory **must** contain at minimum:

| File                     | Purpose                                                                      |
| ------------------------ | ---------------------------------------------------------------------------- |
| `resolved_config.yaml` | Full Hydra config after all defaults and overrides                           |
| `metadata.json`        | Timestamp (ISO), git hash, seed, CLI overrides, package versions             |
| `metrics.csv`          | Per-episode metrics (episode_reward, coop_rate, opp_coop_rate, trust_margin) |

Agent checkpoints (`agent_model.json` or `agent_model.pt`) are saved alongside
these files.

### 2.3 How to identify a run

- **When**: `metadata.json → timestamp_iso`
- **Which code**: `metadata.json → git_commit`
- **Which config**: open `resolved_config.yaml`
- **Which overrides**: `metadata.json → hydra_overrides`

---

## 3. Pre-change Checklist

Before pushing **any** change, verify:

1. **Tests pass** — `uv run pytest` must exit 0.
2. **Smoke train OK** — `uv run python scripts/train.py train.episodes=2 env.max_rounds=5`
   runs without error and produces a valid experiment folder.
3. **README up to date** — reflects the current state after your change.
4. **Lint clean** — `uv run ruff check src/ scripts/ tests/` reports no errors.

---

## 4. Adding a New Experiment Config

To add a new experiment (e.g. *DQN vs Defector with noise*):

### Step-by-step

1. Create `configs/experiment/dqn_vs_defector_noisy.yaml`:

```yaml
# @package _global_
defaults:
  - override /agent: dqn
  - override /opponent: axelrod_defector

env:
  noise: 0.05

train:
  episodes: 500

seed: 7
```

2. Run it:

```bash
uv run python scripts/train.py experiment=dqn_vs_defector_noisy
```

3. Update the Hydra configuration table in `README.md` to list the new
   experiment.
4. Commit all three changes (YAML + README + any code) together.

### Rules for experiment YAML files

- First line **must** be `# @package _global_`.
- Use `override /group: value` for any config group you want to change.
- Set scalar overrides (episodes, seed, noise …) directly.
- Do **not** duplicate values that already match the defaults — only state
  deltas.

---

## 5. Code Conventions

- **Package**: `ipd_marl` lives under `src/` (src-layout).
- **Agents**: all inherit from `BaseAgent` and implement `act`, `observe`,
  `save`, `load`.
- **No placeholders**: every function must have a real implementation.  No
  `TODO`, `TBD`, `pass`-only bodies, or empty stubs.
- **Type hints**: use them on all public function signatures.
- **Docstrings**: NumPy-style, at least for public classes and functions.

---

## 6. Dependency Management

- Runtime and dev dependencies are declared in `pyproject.toml`.
- Install / sync with `uv sync`.
- To add a dependency: `uv add <package>` (runtime) or `uv add --dev <package>`
  (dev only), then update README if the change is user-visible.

---

## 7. Versioning

- Package version lives in `pyproject.toml` (`version = "…"`).
- Bump the version when making a release or a significant milestone.
- The `__version__` string in `src/ipd_marl/__init__.py` must match.

---

## 8. Visualization

- **Style**: All plotting code **must** import and call `ipd_marl.utils.plot_style.set_style()` before creating figures.
- **Automation**: Training scripts must automatically generate plots using `ipd_marl.utils.plotting`.
- **Comparison**: Use `scripts/visualize.py` for comparative analysis.
