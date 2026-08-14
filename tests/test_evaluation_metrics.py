"""Metric-suite tests with hand-checkable values (SPEC E-1, E-2)."""

from __future__ import annotations

import pytest

from origin_ml.evaluation import compute_binary_metrics


class TestBinaryMetrics:
    def test_perfect_predictions(self) -> None:
        metrics = compute_binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.9, 0.8])
        assert metrics.accuracy == 1.0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1 == 1.0
        assert metrics.auroc == 1.0
        assert metrics.confusion.tp == 2
        assert metrics.confusion.tn == 2
        assert metrics.confusion.fp == 0
        assert metrics.confusion.fn == 0

    def test_hand_computed_mixed_case(self) -> None:
        # y:    1    1    0    0
        # pred: 1    0    1    0   (threshold 0.5)
        metrics = compute_binary_metrics([1, 1, 0, 0], [0.9, 0.3, 0.7, 0.1])
        assert metrics.accuracy == 0.5
        assert metrics.precision == pytest.approx(1 / 2)  # tp=1, fp=1
        assert metrics.recall == pytest.approx(1 / 2)  # tp=1, fn=1
        assert metrics.f1 == pytest.approx(0.5)
        # Pairs: (0.9>0.7)✓ (0.9>0.1)✓ (0.3<0.7)✗ (0.3>0.1)✓ → AUROC = 3/4
        assert metrics.auroc == pytest.approx(0.75)
        brier = ((0.9 - 1) ** 2 + (0.3 - 1) ** 2 + (0.7 - 0) ** 2 + (0.1 - 0) ** 2) / 4
        assert metrics.brier == pytest.approx(brier)

    def test_custom_threshold(self) -> None:
        metrics = compute_binary_metrics([1, 0], [0.4, 0.3], threshold=0.35)
        assert metrics.confusion.tp == 1
        assert metrics.confusion.fp == 0
        assert metrics.accuracy == 1.0

    def test_single_class_has_no_auroc(self) -> None:
        metrics = compute_binary_metrics([1, 1], [0.9, 0.8])
        assert metrics.auroc is None
        assert metrics.accuracy == 1.0

    def test_calibration_bins(self) -> None:
        y = [0] * 10 + [1] * 10
        p = [0.05] * 10 + [0.95] * 10
        metrics = compute_binary_metrics(y, p)
        assert len(metrics.calibration_bins) == 2
        low, high = metrics.calibration_bins
        assert low.mean_predicted == pytest.approx(0.05)
        assert low.frac_positive == 0.0
        assert high.mean_predicted == pytest.approx(0.95)
        assert high.frac_positive == 1.0

    def test_input_validation(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            compute_binary_metrics([], [])
        with pytest.raises(ValueError, match="shape mismatch"):
            compute_binary_metrics([1], [0.5, 0.5])
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            compute_binary_metrics([1], [1.5])
