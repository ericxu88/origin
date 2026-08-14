"""Classification and calibration metrics (SPEC E-1, E-2).

All metrics operate on binary ground truth (1 = ai) plus predicted
probabilities, so the same function serves document-level classification and
sentence-level localization. Outputs are pydantic models: experiment results
serialize to JSON verbatim (SPEC E-5).
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import numpy as np
from pydantic import BaseModel, ConfigDict

__all__ = ["CalibrationBin", "ClassificationMetrics", "ConfusionMatrix", "compute_binary_metrics"]


class ConfusionMatrix(BaseModel):
    model_config = ConfigDict(frozen=True)

    tp: int
    fp: int
    tn: int
    fn: int


class CalibrationBin(BaseModel):
    """One reliability-diagram bin: mean predicted vs observed frequency."""

    model_config = ConfigDict(frozen=True)

    mean_predicted: float
    frac_positive: float
    count: int


class ClassificationMetrics(BaseModel):
    """Binary metrics at a decision threshold plus threshold-free AUROC/Brier."""

    model_config = ConfigDict(frozen=True)

    n: int
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    auroc: float | None
    brier: float
    confusion: ConfusionMatrix
    calibration_bins: tuple[CalibrationBin, ...]


def _calibration_bins(y: np.ndarray, p: np.ndarray, n_bins: int) -> tuple[CalibrationBin, ...]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[CalibrationBin] = []
    for lo, hi in pairwise(edges):
        mask = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        count = int(mask.sum())
        if count == 0:
            continue
        bins.append(
            CalibrationBin(
                mean_predicted=float(p[mask].mean()),
                frac_positive=float(y[mask].mean()),
                count=count,
            )
        )
    return tuple(bins)


def compute_binary_metrics(
    y_true: Sequence[int],
    p_pred: Sequence[float],
    *,
    threshold: float = 0.5,
    n_calibration_bins: int = 10,
) -> ClassificationMetrics:
    """Compute the full SPEC E-1 metric suite.

    ``auroc`` is ``None`` when only one class is present (undefined). All
    other metrics use the given threshold; precision/recall/F1 define 0.0 for
    empty denominators.
    """
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(p_pred, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError(f"shape mismatch: y {y.shape} vs p {p.shape}")
    if len(y) == 0:
        raise ValueError("cannot compute metrics on empty inputs")
    if not np.all((p >= 0.0) & (p <= 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")

    pred = (p >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    auroc: float | None = None
    if len(np.unique(y)) == 2:
        from sklearn.metrics import roc_auc_score

        auroc = float(roc_auc_score(y, p))

    return ClassificationMetrics(
        n=len(y),
        threshold=threshold,
        accuracy=float((pred == y).mean()),
        precision=precision,
        recall=recall,
        f1=f1,
        auroc=auroc,
        brier=float(np.mean((p - y) ** 2)),
        confusion=ConfusionMatrix(tp=tp, fp=fp, tn=tn, fn=fn),
        calibration_bins=_calibration_bins(y, p, n_calibration_bins),
    )
