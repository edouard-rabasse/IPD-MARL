"""Retrieve the current Git commit hash (best-effort)."""

from __future__ import annotations

import subprocess


def get_git_hash() -> str:
    """Return the short-ish HEAD commit hash, or ``'no_git'`` on failure.

    Fails gracefully if ``git`` is not installed, the working directory is not
    a repository, or the command times out.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return "no_git"
    return "no_git"
