"""Global RNG seeding for reproducibility."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility.

    Parameters
    ----------
    seed : int
        Non-negative integer seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Do NOT enable deterministic algorithms globally — some ops have no
    # deterministic implementation and would raise at runtime.
