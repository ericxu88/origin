"""Classical baseline detector: calibrated logistic regression (SPEC C-1..C-5).

Training uses scikit-learn; the fitted model is exported to a self-contained
JSON artifact (feature schema, standardization stats, coefficients, Platt
calibration, per-class feature distribution summaries). Inference re-implements
the linear model in numpy directly from the artifact, so:

- artifacts are portable, human-inspectable, and pickle-free (no arbitrary
  code execution on load), and
- ``load()`` works without touching scikit-learn.

Probability semantics: ``predict_proba`` returns ``P(ai)`` — calibrated with
Platt scaling (a 1-D logistic regression on out-of-fold decision scores;
SPEC C-3) when the training config enables it.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = [
    "BaselineConfig",
    "BaselineDetector",
    "DistributionSummary",
    "FeatureImportance",
]

FloatArray = npt.NDArray[np.float64]

_ARTIFACT_KIND = "origin-baseline-logreg"
_FORMAT_VERSION = 1


@dataclass(frozen=True)
class BaselineConfig:
    """Training configuration for the baseline detector."""

    c: float = 1.0
    max_iter: int = 2000
    seed: int = 0
    calibrate: bool = True
    calibration_folds: int = 3


@dataclass(frozen=True)
class DistributionSummary:
    """Per-class summary of one feature's training distribution (SPEC C-5)."""

    mean: float
    std: float
    p05: float
    p25: float
    p50: float
    p75: float
    p95: float

    @classmethod
    def from_values(cls, values: FloatArray) -> DistributionSummary:
        p05, p25, p50, p75, p95 = (float(q) for q in np.percentile(values, [5, 25, 50, 75, 95]))
        return cls(
            mean=float(np.mean(values)),
            std=float(np.std(values)),
            p05=p05,
            p25=p25,
            p50=p50,
            p75=p75,
            p95=p95,
        )


@dataclass(frozen=True)
class FeatureImportance:
    """A feature's standardized logistic-regression coefficient.

    Positive values push toward the AI class. Coefficients apply to
    standardized features, so magnitudes are comparable across features.
    """

    name: str
    coefficient: float


def _sigmoid(x: FloatArray) -> FloatArray:
    return np.asarray(1.0 / (1.0 + np.exp(-x)), dtype=np.float64)


class BaselineDetector:
    """Calibrated logistic regression over engineered features."""

    def __init__(
        self,
        *,
        feature_names: tuple[str, ...],
        scaler_mean: FloatArray,
        scaler_scale: FloatArray,
        coef: FloatArray,
        intercept: float,
        calibration: tuple[float, float] | None,
        feature_distributions: dict[str, dict[str, DistributionSummary]],
        scorer_name: str | None,
        training_meta: dict[str, Any] | None = None,
    ) -> None:
        n = len(feature_names)
        for arr, label in (
            (scaler_mean, "scaler_mean"),
            (scaler_scale, "scaler_scale"),
            (coef, "coef"),
        ):
            if arr.shape != (n,):
                raise ValueError(f"{label} shape {arr.shape} != ({n},)")
        self.feature_names = feature_names
        self._mean = scaler_mean
        self._scale = scaler_scale
        self._coef = coef
        self._intercept = intercept
        self._calibration = calibration
        self.feature_distributions = feature_distributions
        self.scorer_name = scorer_name
        self.training_meta = dict(training_meta or {})

    # ─── Training ────────────────────────────────────────────────────────────

    @classmethod
    def train(
        cls,
        features: FloatArray,
        labels: npt.NDArray[np.int_],
        feature_names: tuple[str, ...],
        *,
        config: BaselineConfig | None = None,
        scorer_name: str | None = None,
        training_meta: dict[str, Any] | None = None,
    ) -> BaselineDetector:
        """Fit on a feature matrix; ``labels`` are 0 = human, 1 = ai (SPEC C-2)."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_predict
        from sklearn.preprocessing import StandardScaler

        config = config or BaselineConfig()
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels, dtype=np.int64)
        if x.ndim != 2 or x.shape[1] != len(feature_names):
            raise ValueError(
                f"feature matrix shape {x.shape} does not match {len(feature_names)} names"
            )
        if set(np.unique(y)) != {0, 1}:
            raise ValueError("labels must contain both classes 0 (human) and 1 (ai)")

        scaler = StandardScaler().fit(x)
        x_std = scaler.transform(x)
        model = LogisticRegression(
            C=config.c, max_iter=config.max_iter, random_state=config.seed
        ).fit(x_std, y)

        calibration: tuple[float, float] | None = None
        if config.calibrate:
            folds = min(config.calibration_folds, int(np.bincount(y).min()))
            if folds >= 2:
                oof_scores = cross_val_predict(
                    LogisticRegression(
                        C=config.c, max_iter=config.max_iter, random_state=config.seed
                    ),
                    x_std,
                    y,
                    cv=folds,
                    method="decision_function",
                )
                platt = LogisticRegression(max_iter=1000).fit(
                    np.asarray(oof_scores, dtype=np.float64).reshape(-1, 1), y
                )
                calibration = (float(platt.coef_[0][0]), float(platt.intercept_[0]))

        # Per-class feature distribution summaries for explainability (C-5).
        distributions: dict[str, dict[str, DistributionSummary]] = {}
        for class_value, class_name in ((0, "human"), (1, "ai")):
            class_rows = x[y == class_value]
            distributions[class_name] = {
                name: DistributionSummary.from_values(class_rows[:, i])
                for i, name in enumerate(feature_names)
            }

        meta = dict(training_meta or {})
        meta.setdefault("n_train", int(x.shape[0]))
        meta.setdefault("config", asdict(config))
        return cls(
            feature_names=feature_names,
            scaler_mean=np.asarray(scaler.mean_, dtype=np.float64),
            scaler_scale=np.asarray(scaler.scale_, dtype=np.float64),
            coef=np.asarray(model.coef_[0], dtype=np.float64),
            intercept=float(model.intercept_[0]),
            calibration=calibration,
            feature_distributions=distributions,
            scorer_name=scorer_name,
            training_meta=meta,
        )

    # ─── Inference (numpy only) ──────────────────────────────────────────────

    def decision_scores(self, features: FloatArray) -> FloatArray:
        x = np.atleast_2d(np.asarray(features, dtype=np.float64))
        if x.shape[1] != len(self.feature_names):
            raise ValueError(f"expected {len(self.feature_names)} features, got {x.shape[1]}")
        scale = np.where(self._scale == 0.0, 1.0, self._scale)
        x_std = (x - self._mean) / scale
        return np.asarray(x_std @ self._coef + self._intercept, dtype=np.float64)

    def predict_proba(self, features: FloatArray) -> FloatArray:
        """Calibrated ``P(ai)`` for one row or a batch (SPEC C-2, C-3)."""
        scores = self.decision_scores(features)
        if self._calibration is not None:
            a, b = self._calibration
            return _sigmoid(a * scores + b)
        return _sigmoid(scores)

    def predict_proba_one(self, features: FloatArray) -> float:
        return float(self.predict_proba(features)[0])

    @property
    def is_calibrated(self) -> bool:
        return self._calibration is not None

    def feature_importances(self) -> list[FeatureImportance]:
        """Standardized coefficients, sorted by absolute magnitude (SPEC C-4)."""
        pairs = [
            FeatureImportance(name=name, coefficient=float(c))
            for name, c in zip(self.feature_names, self._coef, strict=True)
        ]
        return sorted(pairs, key=lambda p: abs(p.coefficient), reverse=True)

    # ─── Serialization (SPEC C-2) ────────────────────────────────────────────

    def save(self, path: Path) -> None:
        payload = {
            "kind": _ARTIFACT_KIND,
            "format_version": _FORMAT_VERSION,
            "feature_names": list(self.feature_names),
            "scaler": {"mean": self._mean.tolist(), "scale": self._scale.tolist()},
            "coef": self._coef.tolist(),
            "intercept": self._intercept,
            "calibration": (
                {"method": "platt", "a": self._calibration[0], "b": self._calibration[1]}
                if self._calibration is not None
                else None
            ),
            "scorer_name": self.scorer_name,
            "training_meta": self.training_meta,
            "feature_distributions": {
                class_name: {name: asdict(summary) for name, summary in class_dists.items()}
                for class_name, class_dists in self.feature_distributions.items()
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> BaselineDetector:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("kind") != _ARTIFACT_KIND:
            raise ValueError(f"{path}: not a baseline artifact (kind={payload.get('kind')!r})")
        if payload.get("format_version") != _FORMAT_VERSION:
            raise ValueError(
                f"{path}: unsupported format version {payload.get('format_version')!r}"
            )
        calibration_raw = payload.get("calibration")
        distributions = {
            class_name: {
                name: DistributionSummary(**summary) for name, summary in class_dists.items()
            }
            for class_name, class_dists in payload["feature_distributions"].items()
        }
        return cls(
            feature_names=tuple(payload["feature_names"]),
            scaler_mean=np.asarray(payload["scaler"]["mean"], dtype=np.float64),
            scaler_scale=np.asarray(payload["scaler"]["scale"], dtype=np.float64),
            coef=np.asarray(payload["coef"], dtype=np.float64),
            intercept=float(payload["intercept"]),
            calibration=(
                (float(calibration_raw["a"]), float(calibration_raw["b"]))
                if calibration_raw is not None
                else None
            ),
            feature_distributions=distributions,
            scorer_name=payload.get("scorer_name"),
            training_meta=payload.get("training_meta", {}),
        )


def sigmoid_scalar(x: float) -> float:
    """Numerically stable scalar sigmoid, exposed for reuse."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)
