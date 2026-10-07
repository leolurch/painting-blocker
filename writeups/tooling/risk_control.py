"""Operating points with a guarantee on identity-weighted pairs completeness.

The loss of painting p at an operating point is one minus the share of its
true pairs that the candidate set keeps. It lies in [0, 1], it does not grow
when the candidate set grows, and its mean over the paintings is 1 - PC_id.
Calibration and test paintings come from one random split, so they are
exchangeable. For a target rho, with alpha = 1 - rho, conformal risk control
(Angelopoulos et al., 2024) fits the operating point: the expected loss of a
new painting is at most alpha.

Operating points are passed from the smallest candidate set to the largest,
so the loss does not increase along the first axis.
"""

from __future__ import annotations

import numpy as np


def conformal_feasible(losses: np.ndarray, alpha: float) -> np.ndarray:
    """Per operating point: does the conformal risk control bound hold (losses bounded by 1)?"""
    x = np.atleast_2d(np.asarray(losses, dtype=np.float64))
    n = x.shape[1]
    return n / (n + 1) * x.mean(axis=1) + 1 / (n + 1) <= alpha


def first_safe(holds: np.ndarray) -> int | None:
    """First operating point from which the bound holds at every larger candidate set, or None."""
    failing = np.flatnonzero(~np.asarray(holds, dtype=bool))
    if failing.size == 0:
        return 0
    if failing[-1] == len(holds) - 1:
        return None
    return int(failing[-1]) + 1


def conformal_paintings_needed(alpha: float) -> int:
    """Calibration paintings for which conformal risk control can reach alpha at all."""
    return int(np.ceil(1 / alpha - 1 - 1e-12))
