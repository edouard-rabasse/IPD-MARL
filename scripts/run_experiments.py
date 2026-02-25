#!/usr/bin/env python
"""Run the full evolution experiment grid.

Executes 10 experiment configs x 3 seeds = 30 runs, recording each run's
output directory in ``experiments/experiment_manifest.json`` for downstream
analysis.

Usage::

    uv run python scripts/run_experiments.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time

# ── Experimental grid ────────────────────────────────────────────────────
EXPERIMENTS: list[str] = [
    # Without Axelrod agents
    "evolution_6t_0d",
    "evolution_4t_2d",
    "evolution_3t_3d",
    "evolution_2t_4d",
    "evolution_0t_6d",
    # With Axelrod agents
    "evolution_6t_0d_axelrod",
    "evolution_4t_2d_axelrod",
    "evolution_3t_3d_axelrod",
    "evolution_2t_4d_axelrod",
    "evolution_0t_6d_axelrod",
]

SEEDS: list[int] = [42, 123, 456]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_PATH = os.path.join(ROOT, "experiments", "experiment_manifest.json")


def _run_one(experiment: str, seed: int) -> str | None:
    """Launch a single evolution run and return its run directory, or None on failure."""
    cmd = [
        sys.executable,
        os.path.join(ROOT, "scripts", "evolution.py"),
        f"experiment={experiment}",
        f"seed={seed}",
    ]

    tag = f"{experiment} / seed={seed}"
    print(f"  >> Starting: {tag}")
    t0 = time.time()

    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=600,  # 10-minute timeout per run
            env=env,
        )
    except subprocess.TimeoutExpired:
        print(f"  [X] TIMEOUT: {tag}")
        return None

    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  [X] FAILED ({elapsed:.1f}s): {tag}")
        # Print last 10 lines of stderr for debugging
        stderr_lines = result.stderr.strip().split("\n")
        for line in stderr_lines[-10:]:
            print(f"    {line}")
        return None

    # Parse run directory from stdout (line: ">> Run directory: ...")
    run_dir = None
    for line in result.stdout.split("\n"):
        m = re.search(r"Run directory:\s*(.+)", line)
        if m:
            run_dir = m.group(1).strip()
            break

    if run_dir:
        print(f"  [OK] Done ({elapsed:.1f}s): {tag} -> {os.path.basename(run_dir)}")
    else:
        print(f"  [WARN] Completed but run dir not found ({elapsed:.1f}s): {tag}")

    return run_dir


def main() -> None:
    """Execute the full experiment grid and save the manifest."""
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)

    # Load existing manifest if present (to allow resuming)
    manifest: dict[str, dict[str, str | None]] = {}
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            manifest = json.load(f)

    total = len(EXPERIMENTS) * len(SEEDS)
    completed = 0
    failed = 0

    print(f"=== Evolution Experiment Grid: {total} runs ===\n")

    for experiment in EXPERIMENTS:
        print(f"\n-- Condition: {experiment} --")
        for seed in SEEDS:
            key = f"{experiment}__seed{seed}"

            # Skip if already completed
            if key in manifest and manifest[key].get("run_dir"):
                run_dir = manifest[key]["run_dir"]
                if os.path.isdir(run_dir):
                    print(f"  [SKIP] Skipping (already done): {key}")
                    completed += 1
                    continue

            run_dir = _run_one(experiment, seed)

            manifest[key] = {
                "experiment": experiment,
                "seed": seed,
                "run_dir": run_dir,
            }

            # Save manifest after each run (crash-resilient)
            with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

            if run_dir:
                completed += 1
            else:
                failed += 1

    print(f"\n=== Grid complete: {completed} succeeded, {failed} failed ===")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
