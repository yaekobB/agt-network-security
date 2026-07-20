"""
src/utils/seed.py

Reproducibility helpers: set seeds for random generators.
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np


def set_seed(seed: Optional[int]) -> None:
    """
    Set RNG seeds for reproducibility.

    Args:
        seed: integer seed. If None, nothing is set.
    """
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
