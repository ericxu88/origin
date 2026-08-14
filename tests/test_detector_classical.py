"""Baseline detector tests (SPEC C-1..C-5)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from origin_ml.datasets import assign_splits, read_jsonl
from origin_ml.detectors import BaselineConfig, BaselineDetector
from origin_ml.features import build_default_pipeline
from origin_ml.scoring import StubScorer
from origin_ml.training import extract_feature_matrix, train_baseline

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample" / "documents.jsonl"
NAMES = ("f_signal", "f_noise")


def separable_data(n: int = 120, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Class 1 has higher f_signal; f_noise is uninformative."""
    rng = np.random.default_rng(seed)
    x0 = np.column_stack([rng.normal(0.0, 1.0, n // 2), rng.normal(0.0, 1.0, n // 2)])
    x1 = np.column_stack([rng.normal(3.0, 1.0, n // 2), rng.normal(0.0, 1.0, n // 2)])
    x = np.vstack([x0, x1])
    y = np.concatenate([np.zeros(n // 2, dtype=np.int64), np.ones(n // 2, dtype=np.int64)])
    return x, y


def train_separable(**kwargs: object) -> BaselineDetector:
    x, y = separable_data()
    return BaselineDetector.train(x, y, NAMES, **kwargs)  # type: ignore[arg-type]


class TestTraining:
    def test_learns_separable_data(self) -> None:
        detector = train_separable()
        x_test, y_test = separable_data(seed=99)
        probs = detector.predict_proba(x_test)
        accuracy = float(np.mean((probs > 0.5) == (y_test == 1)))
        assert accuracy > 0.9

    def test_probabilities_are_valid(self) -> None:
        detector = train_separable()
        x, _ = separable_data(seed=5)
        probs = detector.predict_proba(x)
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    def test_calibration_enabled_by_default(self) -> None:
        assert train_separable().is_calibrated

    def test_calibration_can_be_disabled(self) -> None:
        detector = train_separable(config=BaselineConfig(calibrate=False))
        assert not detector.is_calibrated

    def test_deterministic_given_seed(self) -> None:
        a = train_separable(config=BaselineConfig(seed=7))
        b = train_separable(config=BaselineConfig(seed=7))
        x, _ = separable_data(seed=3)
        assert np.allclose(a.predict_proba(x), b.predict_proba(x))

    def test_rejects_single_class(self) -> None:
        x, _ = separable_data()
        with pytest.raises(ValueError, match="both classes"):
            BaselineDetector.train(x, np.zeros(len(x), dtype=np.int64), NAMES)

    def test_rejects_shape_mismatch(self) -> None:
        x, y = separable_data()
        with pytest.raises(ValueError, match="does not match"):
            BaselineDetector.train(x, y, ("only_one",))


class TestImportances:
    def test_signal_feature_dominates(self) -> None:
        importances = train_separable().feature_importances()
        assert importances[0].name == "f_signal"
        assert importances[0].coefficient > 0  # pushes toward AI class
        assert abs(importances[0].coefficient) > abs(importances[1].coefficient)

    def test_covers_all_features(self) -> None:
        assert {i.name for i in train_separable().feature_importances()} == set(NAMES)


class TestDistributions:
    def test_embedded_per_class_summaries(self) -> None:
        detector = train_separable()
        assert set(detector.feature_distributions) == {"human", "ai"}
        human = detector.feature_distributions["human"]["f_signal"]
        ai = detector.feature_distributions["ai"]["f_signal"]
        assert ai.mean > human.mean + 2.0  # classes were built 3 sigma apart
        assert human.p25 < human.p50 < human.p75


class TestSerialization:
    def test_round_trip_predictions_identical(self, tmp_path: Path) -> None:
        detector = train_separable()
        path = tmp_path / "baseline.json"
        detector.save(path)
        loaded = BaselineDetector.load(path)
        x, _ = separable_data(seed=11)
        assert np.allclose(detector.predict_proba(x), loaded.predict_proba(x))
        assert loaded.feature_names == detector.feature_names
        assert loaded.is_calibrated == detector.is_calibrated
        assert loaded.feature_distributions == detector.feature_distributions
        assert loaded.training_meta["n_train"] == 120

    def test_rejects_wrong_kind(self, tmp_path: Path) -> None:
        path = tmp_path / "bogus.json"
        path.write_text('{"kind": "other"}', encoding="utf-8")
        with pytest.raises(ValueError, match="not a baseline artifact"):
            BaselineDetector.load(path)


class TestOnSampleCorpus:
    """End-to-end: sample records -> features -> trained detector (SPEC C-2)."""

    def test_train_and_evaluate_on_sample_data(self) -> None:
        records = assign_splits(read_jsonl(SAMPLE), seed=0)
        pipeline = build_default_pipeline(scorer=StubScorer())
        detector = train_baseline(records, pipeline, dataset_name="sample")

        test_records = [
            r for r in records if r.split == "test" and r.label.value in ("human", "ai")
        ]
        assert test_records
        features = extract_feature_matrix(test_records, pipeline)
        probs = detector.predict_proba(features)
        labels = np.asarray([1 if r.label.value == "ai" else 0 for r in test_records])
        accuracy = float(np.mean((probs > 0.5) == (labels == 1)))
        # The fixture corpus is separable by design; a working pipeline clears this easily.
        assert accuracy > 0.8
        assert detector.scorer_name is not None
        assert detector.training_meta["dataset"] == "sample"

    def test_requires_training_records(self) -> None:
        with pytest.raises(ValueError, match="no pure human/ai records"):
            train_baseline([], build_default_pipeline(scorer=StubScorer()))
