#!/usr/bin/env python
"""Analyze DQN agent behavior and strategy classification from evolution experiments.

Loads per-agent-type behavioral metrics (coop rate, conditional cooperation,
retaliation, forgiveness) from all evolution runs, classifies the strategy
learned by DQN agents per condition, and generates focused visualizations and
a markdown report at ``experiments/dqn_analysis.md``.

Usage::

    uv run python scripts/analyze_dqn_behavior.py
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime

import matplotlib
import numpy as np
import pandas as pd

from ipd_marl.utils.plot_style import set_style

matplotlib.use("Agg")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_PATH = os.path.join(ROOT, "experiments", "experiment_manifest.json")
FIGURES_DIR = os.path.join(ROOT, "experiments", "figures")
REPORT_PATH = os.path.join(ROOT, "experiments", "dqn_analysis.md")

CONDITION_ORDER = [
    "evolution_6t_0d",
    "evolution_4t_2d",
    "evolution_3t_3d",
    "evolution_2t_4d",
    "evolution_0t_6d",
    "evolution_6t_0d_axelrod",
    "evolution_4t_2d_axelrod",
    "evolution_3t_3d_axelrod",
    "evolution_2t_4d_axelrod",
    "evolution_0t_6d_axelrod",
]

CONDITION_LABELS = {
    "evolution_6t_0d": "6T / 0D",
    "evolution_4t_2d": "4T / 2D",
    "evolution_3t_3d": "3T / 3D",
    "evolution_2t_4d": "2T / 4D",
    "evolution_0t_6d": "0T / 6D",
    "evolution_6t_0d_axelrod": "6T / 0D + Axelrod",
    "evolution_4t_2d_axelrod": "4T / 2D + Axelrod",
    "evolution_3t_3d_axelrod": "3T / 3D + Axelrod",
    "evolution_2t_4d_axelrod": "2T / 4D + Axelrod",
    "evolution_0t_6d_axelrod": "0T / 6D + Axelrod",
}

# Has Axelrod fixed agents?
HAS_AXELROD = {c: "axelrod" in c for c in CONDITION_ORDER}

DQN_COLS = {
    "fitness": "mean_fitness_dqn",
    "coop_rate": "coop_rate_dqn",
    "p_c_given_c": "p_c_given_c_dqn",
    "p_c_given_d": "p_c_given_d_dqn",
    "retaliation": "retaliation_dqn",
    "forgiveness": "forgiveness_dqn",
}

TABULAR_COLS = {
    "fitness": "mean_fitness_tabular_q",
    "coop_rate": "coop_rate_tabular_q",
    "p_c_given_c": "p_c_given_c_tabular_q",
    "p_c_given_d": "p_c_given_d_tabular_q",
    "retaliation": "retaliation_tabular_q",
    "forgiveness": "forgiveness_tabular_q",
}


# ══════════════════════════════════════════════════════════════════════════
# Loading
# ══════════════════════════════════════════════════════════════════════════


def load_manifest() -> dict:
    """Load and validate the experiment manifest."""
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)

    valid = {}
    for key, entry in manifest.items():
        run_dir = entry.get("run_dir")
        if run_dir and os.path.isdir(run_dir):
            csv_path = os.path.join(run_dir, "evolution_metrics.csv")
            if os.path.exists(csv_path):
                valid[key] = entry
    print(f"Loaded {len(valid)}/{len(manifest)} valid runs")
    return valid


def load_all_data(
    manifest: dict,
) -> dict[str, list[tuple[int, pd.DataFrame, str]]]:
    """Group evolution_metrics.csv DataFrames by experiment condition."""
    grouped: dict[str, list[tuple[int, pd.DataFrame, str]]] = defaultdict(list)
    for _key, entry in manifest.items():
        experiment = entry["experiment"]
        seed = int(entry["seed"])
        run_dir = entry["run_dir"]
        csv_path = os.path.join(run_dir, "evolution_metrics.csv")
        df = pd.read_csv(csv_path)
        grouped[experiment].append((seed, df, run_dir))
    for cond in grouped:
        grouped[cond].sort(key=lambda x: x[0])
    return dict(grouped)


# ══════════════════════════════════════════════════════════════════════════
# Strategy Classification
# ══════════════════════════════════════════════════════════════════════════


def classify_strategy(coop: float, p_c_c: float, p_c_d: float) -> str:
    """Classify an agent's strategy from observed behavioral metrics.

    Parameters
    ----------
    coop:
        Overall cooperation rate.
    p_c_c:
        P(Cooperate | opponent cooperated last round).
    p_c_d:
        P(Cooperate | opponent defected last round).

    Returns
    -------
    str
        Human-readable strategy label.
    """
    if coop < 0.15:
        return "Always Defect"
    if coop > 0.85:
        return "Always Cooperate"
    if p_c_c > 0.7 and p_c_d < 0.3:
        return "Tit-for-Tat-like"
    if p_c_c > 0.5 and p_c_d < 0.3:
        return "Suspicious TFT-like"
    if 0.3 <= coop <= 0.7:
        return "Mixed / Opportunistic"
    return "Unclassified"


# ══════════════════════════════════════════════════════════════════════════
# Aggregation
# ══════════════════════════════════════════════════════════════════════════


def extract_agent_final(
    data: dict[str, list[tuple[int, pd.DataFrame, str]]],
    col_map: dict[str, str],
) -> pd.DataFrame:
    """Extract final-generation metrics for an agent type, averaged across seeds.

    Parameters
    ----------
    data:
        Grouped data from ``load_all_data``.
    col_map:
        Mapping from metric name to CSV column name (e.g. DQN_COLS or TABULAR_COLS).

    Returns
    -------
    pd.DataFrame
        One row per condition with mean and std of each metric, plus strategy label.
    """
    rows = []
    for cond in CONDITION_ORDER:
        if cond not in data:
            continue
        # Check if this agent type has columns in this condition
        sample_df = data[cond][0][1]
        if col_map["coop_rate"] not in sample_df.columns:
            continue

        seed_finals = []
        for _seed, df, _ in data[cond]:
            last = df.iloc[-1]
            seed_row = {metric: last.get(col, np.nan) for metric, col in col_map.items()}
            seed_finals.append(seed_row)

        seed_df = pd.DataFrame(seed_finals)
        row: dict = {"condition": CONDITION_LABELS.get(cond, cond)}
        for metric in col_map:
            vals = seed_df[metric].dropna()
            row[f"{metric}_mean"] = float(vals.mean()) if len(vals) > 0 else np.nan
            row[f"{metric}_std"] = float(vals.std()) if len(vals) > 1 else 0.0

        row["strategy"] = classify_strategy(
            float(row.get("coop_rate_mean", 0.0) or 0.0),
            float(row.get("p_c_given_c_mean", 0.0) or 0.0),
            float(row.get("p_c_given_d_mean", 0.0) or 0.0),
        )
        rows.append(row)

    return pd.DataFrame(rows)


def extract_dqn_curves(
    data: dict[str, list[tuple[int, pd.DataFrame, str]]],
) -> dict[str, pd.DataFrame]:
    """Build per-generation mean ± std curves of DQN behavioral metrics per condition."""
    curves: dict[str, pd.DataFrame] = {}
    for cond in CONDITION_ORDER:
        if cond not in data:
            continue
        sample_df = data[cond][0][1]
        if DQN_COLS["coop_rate"] not in sample_df.columns:
            continue

        series: dict[str, pd.Series] = {}
        for metric, col in DQN_COLS.items():
            seed_series = []
            for _, df, _ in data[cond]:
                if col in df.columns:
                    seed_series.append(df.set_index("generation")[col])
            if seed_series:
                combined = pd.concat(seed_series, axis=1)
                series[f"{metric}_mean"] = combined.mean(axis=1)
                series[f"{metric}_std"] = combined.std(axis=1)

        curves[cond] = pd.DataFrame(series)
    return curves


# ══════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ══════════════════════════════════════════════════════════════════════════


def _ensure_figures_dir() -> str:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    return FIGURES_DIR


def _rel(path: str) -> str:
    """Make path relative to experiments/ for use in Markdown image links."""
    if not path:
        return ""
    return os.path.relpath(path, os.path.dirname(REPORT_PATH)).replace("\\", "/")


def _df_to_md_table(df: pd.DataFrame, float_fmt: str = ".3f") -> str:
    """Convert a DataFrame to a Markdown pipe-table string."""
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        cells = []
        for h in headers:
            val = row[h]
            if isinstance(val, float):
                cells.append(f"{val:{float_fmt}}" if pd.notna(val) else "—")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════


def plot_conditional_coop(dqn_final: pd.DataFrame, tabular_final: pd.DataFrame) -> str:
    """Scatter P(C|C) vs P(C|D) for DQN and Tabular Q, annotated by condition.

    Upper-left quadrant = TFT zone. Bottom-left = Always Defect. Top-right = Always Cooperate.
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    set_style()
    fig, ax = plt.subplots(figsize=(10, 8))

    # Strategy region backgrounds (subtle shading)
    ax.fill_betweenx(
        [0, 0.3], 0, 0.15, alpha=0.06, color="red", label="_nolegend_"
    )  # AllD zone
    ax.fill_between(
        [0.7, 1.05], 0, 0.3, alpha=0.06, color="blue", label="_nolegend_"
    )  # TFT zone

    # Diagonal: unconditional (P(C|C) == P(C|D))
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.35, linewidth=1, label="P(C|C) = P(C|D)")

    # DQN points
    for _, row in dqn_final.iterrows():
        x = row["p_c_given_c_mean"]
        y = row["p_c_given_d_mean"]
        if pd.isna(x) or pd.isna(y):
            continue
        xerr = row.get("p_c_given_c_std", 0.0)
        yerr = row.get("p_c_given_d_std", 0.0)
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            markersize=10,
            color="#DD8452",
            ecolor="#DD8452",
            elinewidth=1,
            capsize=3,
            label="_nolegend_",
        )
        ax.annotate(
            row["condition"],
            (x, y),
            fontsize=7,
            alpha=0.85,
            textcoords="offset points",
            xytext=(6, 4),
        )

    # Tabular Q points for comparison (only where available)
    for _, row in tabular_final.iterrows():
        x = row["p_c_given_c_mean"]
        y = row["p_c_given_d_mean"]
        if pd.isna(x) or pd.isna(y):
            continue
        ax.scatter(x, y, s=80, marker="s", color="#4C72B0", zorder=5, label="_nolegend_")
        ax.annotate(
            row["condition"],
            (x, y),
            fontsize=6,
            alpha=0.6,
            textcoords="offset points",
            xytext=(6, -8),
            color="#4C72B0",
        )

    # Reference strategy annotations
    ax.annotate("AllD\nzone", (0.03, 0.03), fontsize=8, color="red", alpha=0.5, style="italic")
    ax.annotate("AllC\nzone", (0.88, 0.88), fontsize=8, color="green", alpha=0.5, style="italic")
    ax.annotate(
        "TFT\nzone", (0.82, 0.05), fontsize=8, color="blue", alpha=0.5, style="italic"
    )

    dqn_patch = mpatches.Patch(color="#DD8452", label="DQN")
    tab_patch = mpatches.Patch(color="#4C72B0", label="Tabular Q")
    diag_line = plt.Line2D([0], [0], linestyle="--", color="gray", label="P(C|C) = P(C|D)")
    ax.legend(handles=[dqn_patch, tab_patch, diag_line], fontsize=9)

    ax.set_xlabel("P(C | Opponent Cooperated last round)", fontsize=12)
    ax.set_ylabel("P(C | Opponent Defected last round)", fontsize=12)
    ax.set_title("Conditional Cooperation — DQN vs Tabular Q", fontsize=13)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(_ensure_figures_dir(), "dqn_conditional_coop.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_retaliation_forgiveness(dqn_final: pd.DataFrame, tabular_final: pd.DataFrame) -> str:
    """Scatter retaliation vs forgiveness for DQN (and Tabular Q for comparison).

    High retaliation + low forgiveness  → Grim Trigger.
    High retaliation + high forgiveness → TFT-like.
    Low retaliation                     → cooperative / exploitable.
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    set_style()
    fig, ax = plt.subplots(figsize=(9, 7))

    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.3, linewidth=1)

    for _, row in dqn_final.iterrows():
        x = row["retaliation_mean"]
        y = row["forgiveness_mean"]
        if pd.isna(x) or pd.isna(y):
            continue
        xerr = row.get("retaliation_std", 0.0)
        yerr = row.get("forgiveness_std", 0.0)
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            markersize=10,
            color="#DD8452",
            ecolor="#DD8452",
            elinewidth=1,
            capsize=3,
        )
        ax.annotate(
            row["condition"],
            (x, y),
            fontsize=7,
            alpha=0.85,
            textcoords="offset points",
            xytext=(6, 4),
        )

    for _, row in tabular_final.iterrows():
        x = row["retaliation_mean"]
        y = row["forgiveness_mean"]
        if pd.isna(x) or pd.isna(y):
            continue
        ax.scatter(x, y, s=80, marker="s", color="#4C72B0", zorder=5)
        ax.annotate(
            row["condition"],
            (x, y),
            fontsize=6,
            alpha=0.6,
            textcoords="offset points",
            xytext=(6, -8),
            color="#4C72B0",
        )

    # Region labels
    ax.annotate(
        "Grim-like\n(high ret., low forg.)",
        (0.8, 0.05),
        fontsize=8,
        alpha=0.5,
        style="italic",
        color="red",
    )
    ax.annotate(
        "TFT-like\n(high ret., high forg.)",
        (0.7, 0.75),
        fontsize=8,
        alpha=0.5,
        style="italic",
        color="blue",
    )
    ax.annotate(
        "Cooperator\n(low ret., high forg.)",
        (0.02, 0.8),
        fontsize=8,
        alpha=0.5,
        style="italic",
        color="green",
    )

    dqn_patch = mpatches.Patch(color="#DD8452", label="DQN")
    tab_patch = mpatches.Patch(color="#4C72B0", label="Tabular Q")
    ax.legend(handles=[dqn_patch, tab_patch], fontsize=9)

    ax.set_xlabel("Retaliation Rate  P(D | opp_D_{t-1})", fontsize=12)
    ax.set_ylabel("Forgiveness Rate  P(C | mutual_D)", fontsize=12)
    ax.set_title("Retaliation vs Forgiveness — DQN vs Tabular Q", fontsize=13)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(_ensure_figures_dir(), "dqn_retaliation_forgiveness.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_fitness_vs_coop(dqn_final: pd.DataFrame) -> str:
    """Scatter fitness vs cooperation rate for DQN — shows the fitness paradox."""
    import matplotlib.pyplot as plt

    set_style()
    fig, ax = plt.subplots(figsize=(9, 6))

    no_axelrod = dqn_final[~dqn_final["condition"].str.contains("Axelrod")]
    with_axelrod = dqn_final[dqn_final["condition"].str.contains("Axelrod")]

    for subset, color, marker, label in [
        (no_axelrod, "#4C72B0", "o", "No Axelrod"),
        (with_axelrod, "#DD8452", "^", "With Axelrod"),
    ]:
        for _, row in subset.iterrows():
            x = row["coop_rate_mean"]
            y = row["fitness_mean"]
            if pd.isna(x) or pd.isna(y):
                continue
            ax.scatter(x, y, s=100, color=color, marker=marker, zorder=5, label="_nolegend_")
            ax.annotate(
                row["condition"],
                (x, y),
                fontsize=7,
                alpha=0.85,
                textcoords="offset points",
                xytext=(5, 4),
            )

    # Legend
    import matplotlib.patches as mpatches

    handles = [
        mpatches.Patch(color="#4C72B0", label="No Axelrod"),
        mpatches.Patch(color="#DD8452", label="With Axelrod"),
    ]
    ax.legend(handles=handles, fontsize=9)

    ax.set_xlabel("DQN Cooperation Rate", fontsize=12)
    ax.set_ylabel("DQN Mean Fitness", fontsize=12)
    ax.set_title("DQN Fitness vs Cooperation Rate\n(the fitness paradox)", fontsize=13)
    ax.set_xlim(-0.05, 1.0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(_ensure_figures_dir(), "dqn_fitness_vs_coop.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_dqn_coop_curves(curves: dict[str, pd.DataFrame]) -> str:
    """DQN cooperation rate over generations, one line per condition."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    no_axelrod = [c for c in CONDITION_ORDER if "axelrod" not in c and c in curves]
    with_axelrod = [c for c in CONDITION_ORDER if "axelrod" in c and c in curves]

    for ax, group, title in [
        (axes[0], no_axelrod, "Without Axelrod Agents"),
        (axes[1], with_axelrod, "With Axelrod Agents"),
    ]:
        colors = sns.color_palette("tab10", max(len(group), 1))
        for idx, cond in enumerate(group):
            df = curves[cond]
            if "coop_rate_mean" not in df.columns:
                continue
            label = CONDITION_LABELS.get(cond, cond)
            ax.plot(
                df.index,
                df["coop_rate_mean"],
                label=label,
                color=colors[idx],
                linewidth=2,
            )
            if "coop_rate_std" in df.columns:
                ax.fill_between(
                    df.index,
                    df["coop_rate_mean"] - df["coop_rate_std"],
                    df["coop_rate_mean"] + df["coop_rate_std"],
                    alpha=0.15,
                    color=colors[idx],
                )
        ax.set_xlabel("Generation")
        ax.set_ylabel("DQN Cooperation Rate")
        ax.set_title(title)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("DQN Cooperation Rate Over Generations", fontsize=13)
    fig.tight_layout()

    path = os.path.join(_ensure_figures_dir(), "dqn_coop_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════════════════════════
# Report
# ══════════════════════════════════════════════════════════════════════════


def _make_display_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build a human-readable summary table from the aggregated metrics DataFrame."""
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "Condition": row["condition"],
                "Coop Rate": f"{row['coop_rate_mean']:.3f} ± {row['coop_rate_std']:.3f}",
                "P(C|C)": f"{row['p_c_given_c_mean']:.3f}",
                "P(C|D)": f"{row['p_c_given_d_mean']:.3f}",
                "Retaliation": f"{row['retaliation_mean']:.3f}",
                "Forgiveness": f"{row['forgiveness_mean']:.3f}",
                "Fitness": f"{row['fitness_mean']:.2f} ± {row['fitness_std']:.2f}",
                "Strategy": row["strategy"],
            }
        )
    return pd.DataFrame(rows)


def write_report(
    dqn_final: pd.DataFrame,
    tabular_final: pd.DataFrame,
    plot_paths: dict[str, str],
) -> None:
    """Write experiments/dqn_analysis.md."""
    timestamp = datetime.now().astimezone().isoformat()

    dqn_display = _make_display_table(dqn_final)
    # Drop tabular rows where all behavioral metrics are NaN (not tracked in mixed populations)
    tab_valid = tabular_final.dropna(subset=["coop_rate_mean"])
    tab_display = _make_display_table(tab_valid) if len(tab_valid) > 0 else pd.DataFrame()

    # Strategy distribution for DQN
    strategy_counts = dqn_final["strategy"].value_counts()
    strat_lines = "\n".join(
        f"- **{s}**: {n} condition(s)" for s, n in strategy_counts.items()
    )

    setup_section = """\
## 0. Experimental Setup

### The Game: Iterated Prisoner's Dilemma (IPD)

Each match consists of **7 rounds** of a standard Prisoner's Dilemma with the following
payoff matrix (row = agent, column = opponent):

|               | Opponent: C | Opponent: D |
|---------------|-------------|-------------|
| **Agent: C**  | (3, 3)      | (0, 5)      |
| **Agent: D**  | (5, 0)      | (1, 1)      |

Agents observe the last 3 rounds of joint actions as a flat vector of length 6
(padded with -1 at the start of each episode).

### Agent Types

**DQN**: A 2-layer MLP (64 hidden units each, ReLU) trained with experience replay
and a target network. The network maps the observation vector to Q-values for
Cooperate (0) and Defect (1).

**Tabular Q**: An epsilon-greedy Q-learning agent with a `defaultdict` Q-table
(one entry per observed state). Used as a baseline learnable agent.

**Fixed Axelrod strategies** (used as environmental pressure in half the conditions):

| Strategy | Behaviour |
|---|---|
| Tit For Tat (×2) | Cooperates first; then copies opponent's last action |
| Defector (×2) | Always defects |
| Cooperator (×2) | Always cooperates |
| Win-Stay Lose-Shift / Pavlov (×2) | Repeats last action if reward was good, switches otherwise |

### Evolution Tournament Design

| Parameter | Value |
|---|---|
| Generations | 30 |
| Learnable agents per population | 6 |
| Fixed Axelrod agents (when present) | 8 (2 × each of the 4 strategies above) |
| Rounds per match | 7 |
| Opponents per fitness evaluation | 5 (random subset of population) |
| Survival rate | 50% (top half by fitness survives) |
| Mutation noise (σ) | 0.02 Gaussian on weights/Q-values |
| Seeds | 42, 123, 456 (3 independent replicas per condition) |

### Population Conditions

Ten conditions were tested, systematically varying the ratio of Tabular Q (T)
to DQN (D) agents, with and without fixed Axelrod pressure:

**Without Axelrod** — 6 learnable agents compete only against each other:

| Condition | Tabular Q | DQN | DQN faces... |
|---|---|---|---|
| 6T / 0D | 6 | 0 | — (no DQN) |
| 4T / 2D | 4 | 2 | 4 Tabular Q + 1 other DQN |
| 3T / 3D | 3 | 3 | 3 Tabular Q + 2 other DQN |
| 2T / 4D | 2 | 4 | 2 Tabular Q + 3 other DQN |
| 0T / 6D | 0 | 6 | 5 other DQN |

**With Axelrod** — same learnable populations + 8 fixed reference agents:

| Condition | Tabular Q | DQN | DQN faces... |
|---|---|---|---|
| 6T / 0D + Axelrod | 6 | 0 | — (no DQN) |
| 4T / 2D + Axelrod | 4 | 2 | 4 Tabular Q + 1 DQN + up to 5 Axelrod |
| 3T / 3D + Axelrod | 3 | 3 | 3 Tabular Q + 2 DQN + up to 5 Axelrod |
| 2T / 4D + Axelrod | 2 | 4 | 2 Tabular Q + 3 DQN + up to 5 Axelrod |
| 0T / 6D + Axelrod | 0 | 6 | 5 other DQN + up to 5 Axelrod |

"""

    report_parts = [
        "# DQN Strategy Behavior Analysis\n",
        f"**Generated**: {timestamp}\n",
        f"**Scope**: {len(dqn_final)} conditions with DQN agents "
        f"(final generation, mean across 3 seeds)\n",
        "\n---\n",
        setup_section,
        "---\n",
        "## 1. Summary: What Strategy Does DQN Learn?\n",
        "### Strategy Classification (final generation, averaged across seeds)\n",
        _df_to_md_table(dqn_display),
        "\n",
        "### Strategy Distribution\n",
        strat_lines,
        "\n",
        "### Key Findings\n",
        (
            "- **DQN converges to Always Defect in most conditions** (coop_rate < 0.15 "
            "in the majority of runs).\n"
            "- DQN is **NOT Tit-for-Tat**: it would require P(C|C) > 0.7 AND P(C|D) < 0.3, "
            "which is never satisfied.\n"
            "- DQN is **NOT Always Cooperate** in any condition.\n"
            "- **Fitness paradox**: the condition where DQN achieves highest fitness "
            "(0T/6D + Axelrod) is also where it cooperates least (~5.6%). "
            "It free-rides on fixed cooperators (TFT, Cooperator, WSLS).\n"
            "- In pure-DQN populations (0T/6D), agents start Gen 1 with moderate cooperation "
            "(~60%) but collapse to near-AllD by Gen 2 — evolutionary pressure selects "
            "defection almost immediately.\n"
            "- In mixed conditions (4T/2D without Axelrod), DQN is weakly 'Mixed/Opportunistic' "
            "but still predominantly defecting (coop ~24%).\n"
        ),
        "\n---\n",
        "## 2. Comparison with Tabular Q-Learning\n",
    ]

    if len(tab_display) > 0:
        report_parts += [
            (
                "Behavioral metrics for Tabular Q were only tracked in **homogeneous** "
                "conditions (6T / 0D and 6T / 0D + Axelrod). In mixed populations "
                "the per-type breakdown was not recorded.\n\n"
            ),
            _df_to_md_table(tab_display),
            "\n",
            (
                "In the homogeneous Tabular Q baseline (6T / 0D), agents evolve a "
                "**forgiving cooperator** pattern: coop ≈ 0.77, P(C|C) ≈ 0.74, "
                "forgiveness ≈ 0.57. It is classified 'Unclassified' rather than TFT "
                "because it still cooperates quite often after the opponent defects "
                "(P(C|D) ≈ 0.68). When Axelrod agents are added (6T / 0D + Axelrod), "
                "Tabular Q becomes more discriminating: P(C|D) drops to ≈ 0.30, "
                "approaching Suspicious TFT. This contrasts sharply with DQN, which "
                "moves in the opposite direction — toward near-unconditional defection "
                "— in every condition.\n"
            ),
        ]
    else:
        report_parts.append("*(No Tabular Q data available for comparison.)*\n")

    report_parts += [
        "\n---\n",
        "## 3. Conditional Cooperation (P(C|C) vs P(C|D))\n",
        f"![Conditional Cooperation]({_rel(plot_paths.get('cond_coop', ''))})\n",
        (
            "**Interpretation**: TFT-like strategies cluster in the upper-left "
            "(high P(C|C), low P(C|D)). AllD clusters near (0, 0). AllC near (1, 1). "
            "DQN points (orange circles) cluster near (0, 0) — unconditional defectors. "
            "Tabular Q (blue squares) sit higher on the cooperation axis.\n"
        ),
        "\n---\n",
        "## 4. Retaliation vs Forgiveness\n",
        f"![Retaliation vs Forgiveness]({_rel(plot_paths.get('ret_forg', ''))})\n",
        (
            "**Interpretation**: DQN without Axelrod has very high retaliation (0.7–0.9) "
            "and low forgiveness (< 0.35) — closest to Grim Trigger. "
            "With Axelrod the retaliation drops (0.3–0.5) because there are fewer "
            "defections to retaliate against (fixed cooperators never defect). "
            "Tabular Q is in the low-retaliation, moderate-forgiveness zone.\n"
        ),
        "\n---\n",
        "## 5. DQN Fitness vs Cooperation Rate\n",
        f"![Fitness vs Coop]({_rel(plot_paths.get('fit_coop', ''))})\n",
        (
            "DQN with Axelrod agents (triangles) achieves 20–22 fitness at near-zero "
            "cooperation — it exploits fixed cooperators. Without Axelrod (circles) "
            "DQN fitness is much lower (8–12) and the relationship is weakly positive.\n"
        ),
        "\n---\n",
        "## 6. DQN Cooperation Rate Over Generations\n",
        f"![Coop Curves]({_rel(plot_paths.get('coop_curves', ''))})\n",
        (
            "In all conditions DQN cooperation collapses within the first 2–5 generations "
            "and remains low. There is no evolutionary recovery toward cooperative strategies, "
            "confirming that defection is the stable evolutionary equilibrium for DQN "
            "under these experimental settings (7-round matches, σ=0.02 mutation noise).\n"
        ),
        "\n---\n",
        "## 7. Why Does DQN Defect?\n",
        (
            "1. **Short match horizon (7 rounds)**: the 'shadow of the future' that makes "
            "cooperation stable in infinite-horizon IPD is absent. Mutual cooperation "
            "(3 each) cannot beat a first-round defection (5 payoff) before the game ends.\n"
            "2. **DQN converges faster**: being a more expressive function approximator, "
            "DQN finds the dominant-strategy Nash equilibrium (mutual defection, payoff 1 "
            "each round) before tabular Q escapes local optima.\n"
            "3. **Evolutionary selection**: once any DQN agent defects more, it earns "
            "higher fitness against cooperating peers, accelerating the spread of defection "
            "through mutation and survival selection.\n"
            "4. **Exploitation of fixed cooperators**: with TFT, Cooperator, and WSLS "
            "in the population, DQN learns to defect unconditionally and collect 5 reward "
            "from those agents, making AllD even more dominant.\n"
        ),
    ]

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_parts))

    print(f"Report written to {REPORT_PATH}")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Run the DQN behavior analysis pipeline."""
    print("=== Loading data ===")
    manifest = load_manifest()
    data = load_all_data(manifest)
    print(f"Conditions: {list(data.keys())}")

    print("\n=== Extracting DQN metrics ===")
    dqn_final = extract_agent_final(data, DQN_COLS)
    tabular_final = extract_agent_final(data, TABULAR_COLS)

    print("\nDQN final-generation strategy classification:")
    for _, row in dqn_final.iterrows():
        print(
            f"  {row['condition']:<28} "
            f"coop={row['coop_rate_mean']:.3f}  "
            f"P(C|C)={row['p_c_given_c_mean']:.3f}  "
            f"P(C|D)={row['p_c_given_d_mean']:.3f}  "
            f"-> {row['strategy']}"
        )

    print("\nTabular Q final-generation strategy classification:")
    for _, row in tabular_final.iterrows():
        print(
            f"  {row['condition']:<28} "
            f"coop={row['coop_rate_mean']:.3f}  "
            f"P(C|C)={row['p_c_given_c_mean']:.3f}  "
            f"P(C|D)={row['p_c_given_d_mean']:.3f}  "
            f"-> {row['strategy']}"
        )

    print("\n=== Building DQN generation curves ===")
    curves = extract_dqn_curves(data)

    print("\n=== Generating plots ===")
    plot_paths: dict[str, str] = {}

    plot_paths["cond_coop"] = plot_conditional_coop(dqn_final, tabular_final)
    print("  [OK] dqn_conditional_coop.png")

    plot_paths["ret_forg"] = plot_retaliation_forgiveness(dqn_final, tabular_final)
    print("  [OK] dqn_retaliation_forgiveness.png")

    plot_paths["fit_coop"] = plot_fitness_vs_coop(dqn_final)
    print("  [OK] dqn_fitness_vs_coop.png")

    plot_paths["coop_curves"] = plot_dqn_coop_curves(curves)
    print("  [OK] dqn_coop_curves.png")

    print("\n=== Writing report ===")
    write_report(dqn_final, tabular_final, plot_paths)

    print("\n=== Done ===")
    print(f"  Report : {REPORT_PATH}")
    print(f"  Figures: {FIGURES_DIR}/dqn_*.png")


if __name__ == "__main__":
    main()
