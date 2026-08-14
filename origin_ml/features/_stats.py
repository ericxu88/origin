"""Small numeric helpers shared by feature extractors.

All statistics are population statistics (``ddof=0``) so that values are
defined (as 0.0) even for a single observation; degenerate inputs never
produce NaN, keeping downstream classifiers safe (SPEC F-12 note on
degeneracy handling).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

__all__ = ["burstiness", "coeff_variation", "safe_max", "safe_mean", "safe_min", "safe_pstd"]


def safe_mean(values: Sequence[float], default: float = 0.0) -> float:
    return sum(values) / len(values) if values else default


def safe_pstd(values: Sequence[float]) -> float:
    """Population standard deviation; 0.0 for fewer than two observations."""
    if len(values) < 2:
        return 0.0
    mu = safe_mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / len(values))


def safe_min(values: Sequence[float], default: float = 0.0) -> float:
    return min(values) if values else default


def safe_max(values: Sequence[float], default: float = 0.0) -> float:
    return max(values) if values else default


def coeff_variation(values: Sequence[float]) -> float:
    """Coefficient of variation ``sigma / mu``; 0.0 when the mean is 0."""
    mu = safe_mean(values)
    if mu == 0.0:
        return 0.0
    return safe_pstd(values) / abs(mu)


def burstiness(values: Sequence[float]) -> float:
    """Goh–Barabási burstiness index ``B = (sigma - mu) / (sigma + mu)``.

    ``B`` is -1 for perfectly regular sequences, ~0 for Poisson-like
    variation, and approaches +1 for highly bursty sequences. Defined as 0.0
    when ``sigma + mu == 0`` (e.g. empty or all-zero input).
    """
    mu = safe_mean(values)
    sigma = safe_pstd(values)
    denom = sigma + mu
    if denom == 0.0:
        return 0.0
    return (sigma - mu) / denom
