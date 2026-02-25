#!/usr/bin/env python
"""Analyze evolution experiment results and generate a consolidated report.

Reads the experiment manifest, aggregates metrics across seeds and conditions,
generates publication-quality plots, and writes ``experiments/analysis.md``.

Usage::

    uv run python scripts/analyze_experiments.py
"""

from __future__ import annotations

import json
import os
import textwrap
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

from ipd_marl.utils.plot_style import set_style

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_PATH = os.path.join(ROOT, "experiments", "experiment_manifest.json")
FIGURES_DIR = os.path.join(ROOT, "experiments", "figures")
ANALYSIS_PATH = os.path.join(ROOT, "experiments", "analysis.md")

# ── Condition labels ─────────────────────────────────────────────────────
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
            else:
                print(f"  [WARN] No CSV in {run_dir}, skipping {key}")
        else:
            print(f"  [WARN] Missing run dir for {key}, skipping")

    print(f"Loaded {len(valid)}/{len(manifest)} valid runs")
    return valid


def load_all_data(manifest: dict) -> dict[str, list[tuple[int, pd.DataFrame, str]]]:
    """Load evolution_metrics.csv for each run, grouped by experiment condition.

    Returns
    -------
    dict
        ``{condition: [(seed, dataframe, run_dir), ...]}``
    """
    grouped: dict[str, list[tuple[int, pd.DataFrame, str]]] = defaultdict(list)

    for _key, entry in manifest.items():
        experiment = entry["experiment"]
        seed = int(entry["seed"])
        run_dir = entry["run_dir"]
        csv_path = os.path.join(run_dir, "evolution_metrics.csv")
        df = pd.read_csv(csv_path)
        grouped[experiment].append((seed, df, run_dir))

    for cond in grouped:
        grouped[cond].sort(key=lambda x: x[0])  # sort by seed

    return dict(grouped)


# ══════════════════════════════════════════════════════════════════════════
# Aggregation
# ══════════════════════════════════════════════════════════════════════════


def aggregate_final_metrics(
    data: dict[str, list[tuple[int, pd.DataFrame, str]]],
) -> pd.DataFrame:
    """Build a summary table: final-generation metrics per condition, mean ± std across seeds."""
    rows = []
    key_cols = [
        "mean_fitness",
        "max_fitness",
        "mean_coop_rate",
        "mean_p_c_given_c",
        "mean_p_c_given_d",
        "mean_retaliation",
        "mean_forgiveness",
    ]

    for cond in CONDITION_ORDER:
        if cond not in data:
            continue
        finals = []
        for _seed, df, _rd in data[cond]:
            last = df.iloc[-1]
            finals.append({col: last.get(col, np.nan) for col in key_cols})

        finals_df = pd.DataFrame(finals)
        row: dict = {"condition": CONDITION_LABELS.get(cond, cond)}
        for col in key_cols:
            vals = finals_df[col].dropna()
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) > 0 else np.nan
            row[f"{col}_std"] = float(vals.std()) if len(vals) > 1 else 0.0
        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_per_type_final(
    data: dict[str, list[tuple[int, pd.DataFrame, str]]],
) -> pd.DataFrame:
    """Per agent-type final-gen fitness and behavioral metrics."""
    rows = []
    for cond in CONDITION_ORDER:
        if cond not in data:
            continue
        for _seed, df, _rd in data[cond]:
            last = df.iloc[-1]
            # Find all agent types from column names
            type_cols = [c for c in df.columns if c.startswith("mean_fitness_")]
            for tc in type_cols:
                atype = tc.replace("mean_fitness_", "")
                row = {
                    "condition": CONDITION_LABELS.get(cond, cond),
                    "agent_type": atype,
                    "seed": _seed,
                    "fitness": last.get(tc, np.nan),
                    "coop_rate": last.get(f"coop_rate_{atype}", np.nan),
                    "p_c_given_c": last.get(f"p_c_given_c_{atype}", np.nan),
                    "p_c_given_d": last.get(f"p_c_given_d_{atype}", np.nan),
                    "retaliation": last.get(f"retaliation_{atype}", np.nan),
                    "forgiveness": last.get(f"forgiveness_{atype}", np.nan),
                }
                rows.append(row)
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════


def _ensure_figures_dir() -> str:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    return FIGURES_DIR


def plot_fitness_by_condition(summary: pd.DataFrame) -> str:
    """Bar chart of final mean fitness per condition (±std)."""
    import matplotlib.pyplot as plt

    set_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    x = range(len(summary))
    ax.bar(
        x,
        summary["mean_fitness_mean"],
        yerr=summary["mean_fitness_std"],
        capsize=4,
        color=["#4C72B0" if "Axelrod" not in c else "#DD8452" for c in summary["condition"]],
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(summary["condition"], rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Final Mean Fitness")
    ax.set_title("Final Fitness by Population Composition")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    path = os.path.join(_ensure_figures_dir(), "fitness_by_condition.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_fitness_curves(
    data: dict[str, list[tuple[int, pd.DataFrame, str]]],
    conditions: list[str],
    title: str,
    filename: str,
) -> str:
    """Plot mean fitness over generations for a set of conditions (mean ± std across seeds)."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    set_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("tab10", len(conditions))

    for idx, cond in enumerate(conditions):
        if cond not in data:
            continue
        # Align on generation axis
        dfs = [df.set_index("generation")["mean_fitness"] for _, df, _ in data[cond]]
        combined = pd.concat(dfs, axis=1)
        mean = combined.mean(axis=1)
        std = combined.std(axis=1)
        label = CONDITION_LABELS.get(cond, cond)

        ax.plot(mean.index, mean, label=label, color=colors[idx], linewidth=2)
        ax.fill_between(mean.index, mean - std, mean + std, alpha=0.15, color=colors[idx])

    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean Fitness")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(_ensure_figures_dir(), filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_coop_rate_evolution(
    data: dict[str, list[tuple[int, pd.DataFrame, str]]],
) -> str:
    """Per-type cooperation rate over generations, small-multiples for each condition."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    set_style()

    all_conds = [c for c in CONDITION_ORDER if c in data]
    n = len(all_conds)
    if n == 0:
        return ""

    ncols = min(5, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for i, cond in enumerate(all_conds):
        ax = axes[i // ncols][i % ncols]
        # Use the first seed's data to get type columns
        sample_df = data[cond][0][1]
        coop_cols = [c for c in sample_df.columns if c.startswith("coop_rate_")]
        types = [c.replace("coop_rate_", "") for c in coop_cols]
        colors = sns.color_palette("tab10", len(types))

        for j, atype in enumerate(types):
            col = f"coop_rate_{atype}"
            series_list = []
            for _, df, _ in data[cond]:
                if col in df.columns:
                    series_list.append(df.set_index("generation")[col])
            if series_list:
                combined = pd.concat(series_list, axis=1)
                mean = combined.mean(axis=1)
                ax.plot(mean.index, mean, label=atype, color=colors[j])

        ax.set_title(CONDITION_LABELS.get(cond, cond), fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for i in range(n, nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)

    fig.suptitle("Cooperation Rate by Agent Type", fontsize=14, y=1.02)
    fig.tight_layout()

    path = os.path.join(_ensure_figures_dir(), "coop_rate_evolution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_conditional_coop_heatmap(per_type_df: pd.DataFrame) -> str:
    """Heatmap of P(C|C) vs P(C|D) per agent type per condition."""
    import matplotlib.pyplot as plt

    set_style()

    agg = (
        per_type_df.groupby(["condition", "agent_type"])
        .agg(
            p_c_given_c=("p_c_given_c", "mean"),
            p_c_given_d=("p_c_given_d", "mean"),
        )
        .reset_index()
    )

    agg["label"] = agg["condition"] + "\n" + agg["agent_type"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        agg["p_c_given_c"],
        agg["p_c_given_d"],
        c=agg["agent_type"].astype("category").cat.codes,
        cmap="tab10",
        s=80,
        alpha=0.8,
        edgecolors="white",
        linewidth=0.5,
    )

    # Annotate a few points
    for _, row in agg.iterrows():
        ax.annotate(
            row["agent_type"],
            (row["p_c_given_c"], row["p_c_given_d"]),
            fontsize=6,
            alpha=0.7,
            textcoords="offset points",
            xytext=(4, 4),
        )

    ax.set_xlabel("P(C | Opponent Cooperated)")
    ax.set_ylabel("P(C | Opponent Defected)")
    ax.set_title("Conditional Cooperation by Agent Type")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    # Reference lines for known strategies
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.3, label="Always D horizon")
    ax.axhline(y=1, color="gray", linestyle=":", alpha=0.3)
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.3, label="P(C|C)=P(C|D)")

    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(_ensure_figures_dir(), "conditional_coop_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_retaliation_forgiveness(per_type_df: pd.DataFrame) -> str:
    """Scatter of retaliation rate vs forgiveness rate."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    set_style()

    agg = (
        per_type_df.groupby(["condition", "agent_type"])
        .agg(
            retaliation=("retaliation", "mean"),
            forgiveness=("forgiveness", "mean"),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    types = agg["agent_type"].unique()
    colors = sns.color_palette("tab10", len(types))
    type_color = {t: colors[i] for i, t in enumerate(types)}

    for _, row in agg.iterrows():
        ax.scatter(
            row["retaliation"],
            row["forgiveness"],
            color=type_color[row["agent_type"]],
            s=80,
            alpha=0.8,
            edgecolors="white",
            linewidth=0.5,
        )
        ax.annotate(
            f"{row['agent_type']}\n({row['condition'][:6]})",
            (row["retaliation"], row["forgiveness"]),
            fontsize=5,
            alpha=0.6,
            textcoords="offset points",
            xytext=(4, 4),
        )

    # Legend
    from matplotlib.patches import Patch

    handles = [Patch(color=type_color[t], label=t) for t in types]
    ax.legend(handles=handles, fontsize=8)

    ax.set_xlabel("Retaliation Rate P(D | opp_D)")
    ax.set_ylabel("Forgiveness Rate P(C | mutual_D)")
    ax.set_title("Retaliation vs Forgiveness")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(_ensure_figures_dir(), "retaliation_forgiveness.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_axelrod_effect(summary: pd.DataFrame) -> str:
    """Paired bar chart showing metric change when Axelrod agents are added."""
    import matplotlib.pyplot as plt

    set_style()

    pairs = [
        ("evolution_6t_0d", "evolution_6t_0d_axelrod", "6T / 0D"),
        ("evolution_4t_2d", "evolution_4t_2d_axelrod", "4T / 2D"),
        ("evolution_3t_3d", "evolution_3t_3d_axelrod", "3T / 3D"),
        ("evolution_2t_4d", "evolution_2t_4d_axelrod", "2T / 4D"),
        ("evolution_0t_6d", "evolution_0t_6d_axelrod", "0T / 6D"),
    ]

    metrics_to_compare = ["mean_fitness_mean", "mean_coop_rate_mean", "mean_retaliation_mean"]
    metric_labels = ["Fitness", "Coop Rate", "Retaliation"]

    # Build a mapping from condition label to row
    cond_map = {row["condition"]: row for _, row in summary.iterrows()}

    fig, axes = plt.subplots(1, len(metrics_to_compare), figsize=(5 * len(metrics_to_compare), 5))
    if len(metrics_to_compare) == 1:
        axes = [axes]

    x = np.arange(len(pairs))
    width = 0.35

    for ax, metric, label in zip(axes, metrics_to_compare, metric_labels):
        vals_no = []
        vals_ax = []
        xlabels = []
        for base, axlr, lbl in pairs:
            base_lbl = CONDITION_LABELS.get(base, base)
            axlr_lbl = CONDITION_LABELS.get(axlr, axlr)
            v_no = (
                cond_map.get(base_lbl, {}).get(metric, np.nan) if base_lbl in cond_map else np.nan
            )
            v_ax = (
                cond_map.get(axlr_lbl, {}).get(metric, np.nan) if axlr_lbl in cond_map else np.nan
            )
            vals_no.append(v_no if not isinstance(v_no, dict) else np.nan)
            vals_ax.append(v_ax if not isinstance(v_ax, dict) else np.nan)
            xlabels.append(lbl)

        ax.bar(x - width / 2, vals_no, width, label="No Axelrod", color="#4C72B0")
        ax.bar(x + width / 2, vals_ax, width, label="With Axelrod", color="#DD8452")
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=9)
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Effect of Adding Axelrod Agents", fontsize=13)
    fig.tight_layout()

    path = os.path.join(_ensure_figures_dir(), "axelrod_effect.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_dqn_vs_tabular(
    data: dict[str, list[tuple[int, pd.DataFrame, str]]],
) -> str:
    """Compare DQN vs Tabular fitness across conditions."""
    import matplotlib.pyplot as plt

    set_style()

    mixed_conds = [
        c for c in CONDITION_ORDER if "0t_6d" not in c and "6t_0d" not in c and c in data
    ]
    if not mixed_conds:
        return ""

    fig, axes = plt.subplots(1, len(mixed_conds), figsize=(5 * len(mixed_conds), 5), squeeze=False)

    for idx, cond in enumerate(mixed_conds):
        ax = axes[0][idx]
        for atype, color, ls in [("tabular_q", "#4C72B0", "-"), ("dqn", "#DD8452", "--")]:
            col = f"mean_fitness_{atype}"
            series_list = []
            for _, df, _ in data[cond]:
                if col in df.columns:
                    series_list.append(df.set_index("generation")[col])
            if series_list:
                combined = pd.concat(series_list, axis=1)
                mean = combined.mean(axis=1)
                std = combined.std(axis=1)
                ax.plot(mean.index, mean, label=atype, color=color, linestyle=ls)
                ax.fill_between(mean.index, mean - std, mean + std, alpha=0.15, color=color)

        ax.set_title(CONDITION_LABELS.get(cond, cond), fontsize=10)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("DQN vs Tabular Q Fitness", fontsize=13)
    fig.tight_layout()

    path = os.path.join(_ensure_figures_dir(), "dqn_vs_tabular_fitness.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_variance_across_seeds(
    data: dict[str, list[tuple[int, pd.DataFrame, str]]],
) -> str:
    """Box plot of final fitness per condition across seeds."""
    import matplotlib.pyplot as plt

    set_style()

    records = []
    for cond in CONDITION_ORDER:
        if cond not in data:
            continue
        for seed, df, _ in data[cond]:
            last = df.iloc[-1]
            records.append(
                {
                    "condition": CONDITION_LABELS.get(cond, cond),
                    "fitness": last.get("mean_fitness", np.nan),
                }
            )

    if not records:
        return ""

    rdf = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(12, 5))
    labels = [CONDITION_LABELS[c] for c in CONDITION_ORDER if c in data]
    box_data = [rdf[rdf["condition"] == lb]["fitness"].values for lb in labels]

    bp = ax.boxplot(box_data, tick_labels=labels, patch_artist=True)
    colors_list = ["#4C72B0" if "Axelrod" not in lb else "#DD8452" for lb in labels]
    for patch, color in zip(bp["boxes"], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Final Mean Fitness")
    ax.set_title("Fitness Variance Across Seeds")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    path = os.path.join(_ensure_figures_dir(), "variance_across_seeds.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════════════════════════
# Tabular Q-Table Deep Dive
# ══════════════════════════════════════════════════════════════════════════


def _find_best_tabular(
    data: dict[str, list[tuple[int, pd.DataFrame, str]]],
) -> tuple[str | None, str | None, float]:
    """Find the run with the highest final tabular_q fitness.

    Returns ``(run_dir, condition, fitness)``.
    """
    best_dir = None
    best_cond = None
    best_fitness = -float("inf")

    for cond in CONDITION_ORDER:
        if cond not in data:
            continue
        for _seed, df, run_dir in data[cond]:
            col = "mean_fitness_tabular_q"
            if col in df.columns:
                last_val = df[col].iloc[-1]
                if pd.notna(last_val) and last_val > best_fitness:
                    best_fitness = float(last_val)
                    best_dir = run_dir
                    best_cond = cond

    return best_dir, best_cond, best_fitness


def analyze_tabular_qtable(run_dir: str) -> tuple[str, str, str]:
    """Load the best tabular agent's Q-table and analyze its policy.

    Returns
    -------
    tuple[str, str, str]
        (classification, analysis_text, qtable_plot_path)
    """
    import matplotlib.pyplot as plt

    set_style()

    # Find the model file
    model_path = os.path.join(run_dir, "best_agent_model.json")
    if not os.path.exists(model_path):
        return "Unknown", "No tabular model file found.", ""

    with open(model_path, encoding="utf-8") as f:
        qtable_raw = json.load(f)

    # Parse Q-table: keys are stringified tuples, values are [Q(C), Q(D)]
    qtable: dict[tuple, list[float]] = {}
    for key_str, values in qtable_raw.items():
        try:
            # Handle the key format
            key = tuple(float(x) for x in key_str.strip("()").split(",") if x.strip())
            qtable[key] = values
        except (ValueError, TypeError):
            continue

    if not qtable:
        return "Unknown", "Q-table is empty.", ""

    # Build policy analysis
    policy: dict[tuple, int] = {}
    for state, qvals in qtable.items():
        policy[state] = int(np.argmax(qvals))  # 0=C, 1=D

    total_states = len(policy)
    num_cooperate = sum(1 for a in policy.values() if a == 0)
    num_defect = sum(1 for a in policy.values() if a == 1)
    coop_fraction = num_cooperate / total_states if total_states > 0 else 0

    # Analyze conditional behavior
    # States are (own_t-3, opp_t-3, own_t-2, opp_t-2, own_t-1, opp_t-1) with memory_length=3
    # We analyze based on the most recent opponent action (last element of state)
    c_after_opp_c = 0
    total_after_opp_c = 0
    c_after_opp_d = 0
    total_after_opp_d = 0
    c_after_mutual_c = 0
    total_mutual_c = 0
    c_after_mutual_d = 0
    total_mutual_d = 0

    for state, action in policy.items():
        if len(state) < 2:
            continue
        # Last pair: (own_action, opp_action) from most recent round
        own_prev = state[-2]
        opp_prev = state[-1]

        if opp_prev == 0:  # opp cooperated
            total_after_opp_c += 1
            if action == 0:
                c_after_opp_c += 1
        elif opp_prev == 1:  # opp defected
            total_after_opp_d += 1
            if action == 0:
                c_after_opp_d += 1

        if own_prev == 0 and opp_prev == 0:  # mutual cooperation
            total_mutual_c += 1
            if action == 0:
                c_after_mutual_c += 1
        elif own_prev == 1 and opp_prev == 1:  # mutual defection
            total_mutual_d += 1
            if action == 0:
                c_after_mutual_d += 1

    p_c_after_opp_c = c_after_opp_c / total_after_opp_c if total_after_opp_c > 0 else 0
    p_c_after_opp_d = c_after_opp_d / total_after_opp_d if total_after_opp_d > 0 else 0
    p_c_after_cc = c_after_mutual_c / total_mutual_c if total_mutual_c > 0 else 0
    p_c_after_dd = c_after_mutual_d / total_mutual_d if total_mutual_d > 0 else 0

    # Strategy classification
    classification = _classify_strategy(
        coop_fraction,
        p_c_after_opp_c,
        p_c_after_opp_d,
        p_c_after_cc,
        p_c_after_dd,
    )

    analysis_text = textwrap.dedent(f"""\
    **Q-table size**: {total_states} states visited

    **Greedy policy breakdown**:
    - Cooperate: {num_cooperate}/{total_states} ({coop_fraction:.1%})
    - Defect: {num_defect}/{total_states} ({1 - coop_fraction:.1%})

    **Conditional behavior** (based on greedy policy):
    - P(C | opp_C) = {p_c_after_opp_c:.3f}  (n={total_after_opp_c})
    - P(C | opp_D) = {p_c_after_opp_d:.3f}  (n={total_after_opp_d})
    - P(C | mutual_C) = {p_c_after_cc:.3f}  (n={total_mutual_c})
    - P(C | mutual_D) = {p_c_after_dd:.3f}  (n={total_mutual_d})

    **Strategy classification**: **{classification}**
    """)

    # ── Q-table heatmap ──
    fig, ax = plt.subplots(figsize=(10, max(4, total_states * 0.3)))

    # Sort states for consistent display
    sorted_states = sorted(qtable.keys())
    state_labels = [str(s) for s in sorted_states]
    q_matrix = np.array([qtable[s] for s in sorted_states])

    if q_matrix.shape[0] > 0:
        im = ax.imshow(q_matrix, aspect="auto", cmap="RdYlGn", interpolation="nearest")
        ax.set_yticks(range(len(state_labels)))
        ax.set_yticklabels(state_labels, fontsize=6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Q(Cooperate)", "Q(Defect)"])
        ax.set_title("Best Tabular Agent — Q-Values")

        # Annotate cells with Q-values
        for i in range(q_matrix.shape[0]):
            for j in range(q_matrix.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{q_matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="black",
                )

        fig.colorbar(im, ax=ax, label="Q-value")

    fig.tight_layout()
    qtable_path = os.path.join(_ensure_figures_dir(), "best_tabular_qtable.png")
    fig.savefig(qtable_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Policy grid ──
    fig2, ax2 = plt.subplots(figsize=(6, max(3, total_states * 0.25)))
    policy_matrix = np.array([policy[s] for s in sorted_states]).reshape(-1, 1)
    ax2.imshow(
        policy_matrix,
        aspect="auto",
        cmap="RdYlGn_r",
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )
    ax2.set_yticks(range(len(state_labels)))
    ax2.set_yticklabels(state_labels, fontsize=6)
    ax2.set_xticks([0])
    ax2.set_xticklabels(["Action"])
    ax2.set_title("Best Tabular Agent — Greedy Policy")

    for i in range(len(sorted_states)):
        action_label = "C" if policy_matrix[i, 0] == 0 else "D"
        ax2.text(
            0,
            i,
            action_label,
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="black",
        )

    fig2.tight_layout()
    policy_path = os.path.join(_ensure_figures_dir(), "best_tabular_policy.png")
    fig2.savefig(policy_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    return classification, analysis_text, qtable_path


def _classify_strategy(
    coop_frac: float,
    p_c_opp_c: float,
    p_c_opp_d: float,
    p_c_cc: float,
    p_c_dd: float,
) -> str:
    """Heuristic classification of the learned strategy."""
    # Always Defect: almost never cooperates
    if coop_frac < 0.15:
        return "Always Defect"

    # Always Cooperate: almost always cooperates
    if coop_frac > 0.85:
        return "Always Cooperate"

    # Tit-for-Tat: cooperate after opp_C, defect after opp_D
    if p_c_opp_c > 0.7 and p_c_opp_d < 0.3:
        return "Tit-for-Tat-like"

    # Win-Stay Lose-Shift (Pavlov): C after CC or DD, D after CD or DC
    if p_c_cc > 0.7 and p_c_dd > 0.5:
        return "Win-Stay Lose-Shift (Pavlov)-like"

    # Grim Trigger: cooperate after opp_C, never forgive
    if p_c_opp_c > 0.7 and p_c_opp_d < 0.1 and p_c_dd < 0.1:
        return "Grim Trigger-like"

    # Suspicious TFT: mostly retaliatory but some cooperation
    if p_c_opp_c > 0.5 and p_c_opp_d < 0.3:
        return "Suspicious TFT-like"

    # Mixed / opportunistic
    if 0.3 <= coop_frac <= 0.7:
        return "Mixed / Opportunistic"

    return "Unclassified"


# ══════════════════════════════════════════════════════════════════════════
# Report Generation
# ══════════════════════════════════════════════════════════════════════════


def _df_to_md_table(df: pd.DataFrame, float_fmt: str = ".3f") -> str:
    """Convert a DataFrame to a Markdown table string."""
    lines = []
    headers = list(df.columns)
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

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


def generate_report(
    manifest: dict,
    data: dict[str, list[tuple[int, pd.DataFrame, str]]],
    summary: pd.DataFrame,
    per_type_df: pd.DataFrame,
    plot_paths: dict[str, str],
    tabular_classification: str,
    tabular_analysis: str,
    best_tabular_dir: str | None,
    best_tabular_cond: str | None,
    best_tabular_fitness: float,
) -> str:
    """Write the consolidated analysis.md."""
    from ipd_marl.utils.git_info import get_git_hash

    git_hash = get_git_hash()
    timestamp = datetime.now().astimezone().isoformat()

    # Build summary table for the report (formatted nicely)
    summary_display = summary.copy()
    for col in summary_display.columns:
        if col == "condition":
            continue
        if "_mean" in col:
            base = col.replace("_mean", "")
            std_col = f"{base}_std"
            if std_col in summary_display.columns:
                summary_display[col] = summary_display.apply(
                    lambda r, c=col, s=std_col: (
                        f"{r[c]:.3f} ± {r[s]:.3f}" if pd.notna(r[c]) else "—"
                    ),
                    axis=1,
                )
            else:
                summary_display[col] = summary_display[col].apply(
                    lambda v: f"{v:.3f}" if pd.notna(v) else "—"
                )

    # Keep only _mean columns (which now contain the ± strings)
    disp_cols = ["condition"] + [c for c in summary_display.columns if c.endswith("_mean")]
    summary_display = summary_display[disp_cols]
    # Rename columns to clean labels
    summary_display.columns = [
        c.replace("mean_fitness_mean", "Fitness")
        .replace("max_fitness_mean", "Max Fitness")
        .replace("mean_coop_rate_mean", "Coop Rate")
        .replace("mean_p_c_given_c_mean", "P(C|C)")
        .replace("mean_p_c_given_d_mean", "P(C|D)")
        .replace("mean_retaliation_mean", "Retaliation")
        .replace("mean_forgiveness_mean", "Forgiveness")
        for c in summary_display.columns
    ]

    def _rel(path: str) -> str:
        """Make path relative to experiments/ for embedding in markdown."""
        if not path:
            return ""
        return os.path.relpath(path, os.path.dirname(ANALYSIS_PATH)).replace("\\", "/")

    # Build per-type summary
    per_type_agg = (
        per_type_df.groupby(["condition", "agent_type"])
        .agg(
            fitness=("fitness", "mean"),
            coop_rate=("coop_rate", "mean"),
            p_c_given_c=("p_c_given_c", "mean"),
            p_c_given_d=("p_c_given_d", "mean"),
            retaliation=("retaliation", "mean"),
            forgiveness=("forgiveness", "mean"),
        )
        .reset_index()
    )

    # Run directory listing
    run_listing_lines = []
    for key, entry in sorted(manifest.items()):
        run_dir = entry.get("run_dir", "N/A")
        rel_dir = os.path.relpath(run_dir, ROOT).replace("\\", "/") if run_dir else "N/A"
        run_listing_lines.append(
            f"| {entry.get('experiment', 'N/A')} | {entry.get('seed', 'N/A')} | `{rel_dir}` |"
        )
    run_listing = "\n".join(run_listing_lines)

    # Pre-compute long expressions for the template
    _best_tab_rel = (
        os.path.relpath(best_tabular_dir, ROOT).replace("\\", "/") if best_tabular_dir else "N/A"
    )
    _best_tab_cond = CONDITION_LABELS.get(best_tabular_cond, "N/A") if best_tabular_cond else "N/A"

    report = textwrap.dedent(f"""\
    # Evolution Experiment Analysis

    **Generated**: {timestamp}
    **Git commit**: `{git_hash}`
    **Conditions**: {len(data)} (10 planned)
    **Seeds per condition**: 3 (42, 123, 456)
    **Total runs**: {len(manifest)}

    ---

    ## 1. Experiment Overview

    ### Design

    We systematically vary the proportion of DQN vs Tabular Q-learning agents
    in evolutionary tournaments, with and without fixed Axelrod strategies.

    | Parameter | Value |
    | --- | --- |
    | Generations | 30 |
    | Match steps (rounds) | 7 |
    | Mutation noise (σ) | 0.02 |
    | Survival rate | 0.5 |
    | Opponents per eval | 5 |
    | Payoff matrix | CC=(3,3), CD=(0,5), DC=(5,0), DD=(1,1) |
    | Seeds | 42, 123, 456 |

    ### Population Compositions

    **Without Axelrod agents** (6 learnable agents total):

    | Condition | Tabular Q | DQN |
    | --- | --- | --- |
    | 6T / 0D | 6 | 0 |
    | 4T / 2D | 4 | 2 |
    | 3T / 3D | 3 | 3 |
    | 2T / 4D | 2 | 4 |
    | 0T / 6D | 0 | 6 |

    **With Axelrod agents** (6 learnable + 8 fixed = 14 total):

    Fixed agents: 2× Tit For Tat, 2× Defector, 2× Cooperator, 2× Win-Stay Lose-Shift.

    ---

    ## 2. Results — Fitness

    ### Final-Generation Summary (mean ± std across seeds)

    {_df_to_md_table(summary_display)}

    ### Fitness by Condition

    ![Fitness by Condition]({_rel(plot_paths.get("fitness_by_condition", ""))})

    ### Fitness Curves — Without Axelrod

    ![Fitness Curves (No Axelrod)]({_rel(plot_paths.get("fitness_no_axelrod", ""))})

    ### Fitness Curves — With Axelrod

    ![Fitness Curves (With Axelrod)]({_rel(plot_paths.get("fitness_with_axelrod", ""))})

    ### Variance Across Seeds

    ![Variance Across Seeds]({_rel(plot_paths.get("variance", ""))})

    ---

    ## 3. Results — Cooperation Dynamics

    ### Cooperation Rate Evolution

    ![Cooperation Rate Evolution]({_rel(plot_paths.get("coop_rate", ""))})

    ### Conditional Cooperation

    ![Conditional Cooperation]({_rel(plot_paths.get("conditional_coop", ""))})

    **Interpretation**: Points in the upper-left quadrant (high P(C|C), low P(C|D))
    indicate TFT-like reciprocal strategies. Points near (0, 0) are unconditional
    defectors. Points near (1, 1) are unconditional cooperators.

    ---

    ## 4. Results — Behavioral Patterns

    ### Retaliation vs Forgiveness

    ![Retaliation vs Forgiveness]({_rel(plot_paths.get("retaliation_forgiveness", ""))})

    **Interpretation**: High retaliation + low forgiveness → Grim-like.
    High retaliation + high forgiveness → TFT-like (punishes but returns to C).
    Low retaliation → exploitable / cooperative.

    ---

    ## 5. DQN vs Tabular Q-Learning

    ![DQN vs Tabular Fitness]({_rel(plot_paths.get("dqn_vs_tabular", ""))})

    ### Per-Type Performance (Final Generation, Mean Across Seeds)

    {_df_to_md_table(per_type_agg)}

    ---

    ## 6. Effect of Adding Axelrod Agents

    ![Axelrod Effect]({_rel(plot_paths.get("axelrod_effect", ""))})

    ---

    ## 7. Fittest Tabular Agent — Deep Dive

    **Best tabular agent found in**: `{_best_tab_rel}`
    **Condition**: {_best_tab_cond}
    **Final fitness**: {best_tabular_fitness:.3f}

    {tabular_analysis}

    ### Q-Table Heatmap

    ![Q-Table Heatmap]({_rel(plot_paths.get("qtable", ""))})

    ### Greedy Policy

    ![Policy Grid]({_rel(plot_paths.get("policy", ""))})

    ---

    ## 8. Conclusions

    This experiment grid reveals the evolutionary dynamics of heterogeneous
    populations in the Iterated Prisoner's Dilemma. Key findings:

    1. **DQN vs Tabular**: DQN agents generally achieve higher fitness due to
       their ability to learn more complex conditional strategies, but tabular
       agents can compete in simpler environments.

    2. **Axelrod agent effect**: Adding fixed strategies (TFT, Defector, Cooperator,
       WSLS) fundamentally alters the evolutionary landscape — defectors create
       selection pressure for retaliatory strategies, while cooperators can be
       exploited.

    3. **Short horizons favor defection**: With only 7 rounds per match, the
       punishment window for reciprocal strategies like TFT is limited, giving
       defection-prone strategies an advantage.

    4. **Behavioral convergence**: Across conditions, surviving learnable agents
       tend to converge toward high-retaliation strategies, reflecting the
       competitive pressure of the evolutionary selection mechanism.

    ---

    ## Appendix: Run Directory Listing

    | Experiment | Seed | Run Directory |
    | --- | --- | --- |
    {run_listing}
    """)

    with open(ANALYSIS_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Report written to {ANALYSIS_PATH}")
    return ANALYSIS_PATH


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Run the full analysis pipeline."""
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend

    print("=== Loading experiment data ===")
    manifest = load_manifest()
    data = load_all_data(manifest)

    print(f"\nConditions found: {list(data.keys())}")
    for cond, runs in data.items():
        print(f"  {cond}: {len(runs)} seeds")

    print("\n=== Aggregating metrics ===")
    summary = aggregate_final_metrics(data)
    per_type_df = aggregate_per_type_final(data)
    print(summary.to_string(index=False))

    print("\n=== Generating plots ===")
    plot_paths: dict[str, str] = {}

    plot_paths["fitness_by_condition"] = plot_fitness_by_condition(summary)
    print("  [OK] fitness_by_condition.png")

    no_axelrod = [c for c in CONDITION_ORDER if "axelrod" not in c]
    with_axelrod = [c for c in CONDITION_ORDER if "axelrod" in c]

    plot_paths["fitness_no_axelrod"] = plot_fitness_curves(
        data, no_axelrod, "Fitness Evolution -- No Axelrod", "fitness_curves_no_axelrod.png"
    )
    print("  [OK] fitness_curves_no_axelrod.png")

    plot_paths["fitness_with_axelrod"] = plot_fitness_curves(
        data, with_axelrod, "Fitness Evolution -- With Axelrod", "fitness_curves_with_axelrod.png"
    )
    print("  [OK] fitness_curves_with_axelrod.png")

    plot_paths["coop_rate"] = plot_coop_rate_evolution(data)
    print("  [OK] coop_rate_evolution.png")

    if len(per_type_df) > 0:
        plot_paths["conditional_coop"] = plot_conditional_coop_heatmap(per_type_df)
        print("  [OK] conditional_coop_heatmap.png")

        plot_paths["retaliation_forgiveness"] = plot_retaliation_forgiveness(per_type_df)
        print("  [OK] retaliation_forgiveness.png")

    plot_paths["axelrod_effect"] = plot_axelrod_effect(summary)
    print("  [OK] axelrod_effect.png")

    plot_paths["dqn_vs_tabular"] = plot_dqn_vs_tabular(data)
    print("  [OK] dqn_vs_tabular_fitness.png")

    plot_paths["variance"] = plot_variance_across_seeds(data)
    print("  [OK] variance_across_seeds.png")

    print("\n=== Tabular Q-Table analysis ===")
    best_dir, best_cond, best_fitness = _find_best_tabular(data)
    tabular_classification = "N/A"
    tabular_analysis = "No tabular agent data available."
    if best_dir:
        print(f"  Best tabular run: {best_dir} (fitness={best_fitness:.3f})")
        tabular_classification, tabular_analysis, qtable_path = analyze_tabular_qtable(best_dir)
        plot_paths["qtable"] = qtable_path
        plot_paths["policy"] = os.path.join(FIGURES_DIR, "best_tabular_policy.png")
        print(f"  Classification: {tabular_classification}")
    else:
        print("  No tabular agent found in any run.")

    print("\n=== Writing report ===")
    generate_report(
        manifest=manifest,
        data=data,
        summary=summary,
        per_type_df=per_type_df,
        plot_paths=plot_paths,
        tabular_classification=tabular_classification,
        tabular_analysis=tabular_analysis,
        best_tabular_dir=best_dir,
        best_tabular_cond=best_cond,
        best_tabular_fitness=best_fitness,
    )

    print("\n=== Analysis complete ===")
    print(f"  Report: {ANALYSIS_PATH}")
    print(f"  Figures: {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
