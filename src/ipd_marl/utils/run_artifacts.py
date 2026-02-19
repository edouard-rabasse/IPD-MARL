"""Run-directory creation and metadata / config persistence."""

from __future__ import annotations

import importlib.metadata
import json
import os
import sys
from datetime import datetime

from omegaconf import DictConfig, OmegaConf

from ipd_marl.utils.git_info import get_git_hash


def make_run_dir(base_dir: str = "experiments", exp_name: str = "run") -> str:
    """Create a unique, timestamped experiment directory.

    Layout::

        <base_dir>/YYYY-MM-DD/HHMMSS_<exp_name>/

    Parameters
    ----------
    base_dir : str
        Root folder for all experiments (default ``"experiments"``).
    exp_name : str
        Human-readable experiment tag appended to the timestamp.

    Returns
    -------
    str
        Absolute path to the newly created directory.
    """
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H%M%S")
    run_name = f"{time_str}_{exp_name}"
    run_path = os.path.join(base_dir, date_str, run_name)
    os.makedirs(run_path, exist_ok=True)
    return os.path.abspath(run_path)


def _pkg_version(name: str) -> str:
    """Return the installed version of *name*, or ``'unknown'``."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def save_run_artifacts(
    run_dir: str,
    cfg: DictConfig,
    overrides: list[str],
    seed: int,
) -> None:
    """Persist the resolved config and rich metadata into *run_dir*.

    Files written:
    - ``resolved_config.yaml``
    - ``metadata.json``
    """
    # 1. Resolved Hydra config
    config_path = os.path.join(run_dir, "resolved_config.yaml")
    with open(config_path, "w", encoding="utf-8") as fh:
        fh.write(OmegaConf.to_yaml(cfg, resolve=True))

    # 2. Metadata
    metadata = {
        "timestamp_iso": datetime.now().astimezone().isoformat(),
        "git_commit": get_git_hash(),
        "hydra_overrides": overrides,
        "seed": seed,
        "python_version": sys.version,
        "package_versions": {
            "numpy": _pkg_version("numpy"),
            "hydra-core": _pkg_version("hydra-core"),
            "axelrod": _pkg_version("axelrod"),
            "torch": _pkg_version("torch"),
        },
    }
    meta_path = os.path.join(run_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)
