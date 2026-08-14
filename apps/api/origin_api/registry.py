"""Detector loading and registry — fully separate from the HTTP layer (SPEC API-6).

Configuration comes from the environment (see ``.env.example``):

- ``ORIGIN_SCORER``            — ``stub`` (default), ``none``, or ``hf:<checkpoint>``.
- ``ORIGIN_ARTIFACT_DIR``      — directory containing trained classical artifacts
  (``doc_baseline.json`` + ``sentence_baseline.json``). When unset, demo
  classical detectors are trained at startup on the bundled sample corpus
  (a few seconds, fully offline) and labelled as such in ``/detectors``.
- ``ORIGIN_NEURAL_CHECKPOINT`` — optional path/hub id of a trained
  sentence-granular neural checkpoint to expose as the ``neural`` detector.
- ``ORIGIN_SAMPLE_DATASET``    — override the demo-training corpus path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from origin_api.schemas import DetectorInfo
from origin_ml.datasets.io import read_jsonl
from origin_ml.datasets.splits import assign_splits
from origin_ml.detectors.classical import BaselineDetector
from origin_ml.detectors.neural import NeuralDetector
from origin_ml.explainability.analyze import AnalysisResult, analyze_document
from origin_ml.features.pipeline import FeaturePipeline, build_default_pipeline
from origin_ml.scoring.base import Scorer
from origin_ml.scoring.stub import StubScorer
from origin_ml.training.classical import train_baseline, train_sentence_baseline

__all__ = ["DetectorRegistry", "LoadedDetector", "make_scorer"]

_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_SAMPLE = _ROOT / "data" / "sample" / "documents.jsonl"


def make_scorer(spec: str) -> Scorer | None:
    if spec == "stub":
        return StubScorer()
    if spec == "none":
        return None
    if spec.startswith("hf:"):
        from origin_ml.scoring.hf import HFCausalScorer

        return HFCausalScorer(spec.removeprefix("hf:"))
    raise ValueError(f"unknown ORIGIN_SCORER {spec!r} (expected 'stub', 'none', or 'hf:<ckpt>')")


@dataclass(frozen=True)
class LoadedDetector:
    """One ready-to-serve detector plus its metadata."""

    name: str
    kind: str  # "classical" | "neural"
    description: str
    source: str
    pipeline: FeaturePipeline
    sentence_baseline: BaselineDetector | None = None
    doc_baseline: BaselineDetector | None = None
    neural: NeuralDetector | None = None
    training_meta: dict[str, str] = field(default_factory=dict)

    def analyze(self, text: str) -> AnalysisResult:
        return analyze_document(
            text,
            pipeline=self.pipeline,
            sentence_baseline=self.sentence_baseline,
            doc_baseline=self.doc_baseline,
            neural=self.neural,
        )

    def info(self) -> DetectorInfo:
        scorer = self.pipeline.scorer
        return DetectorInfo(
            name=self.name,
            kind=self.kind,  # type: ignore[arg-type]
            description=self.description,
            scorer=scorer.name if scorer is not None else None,
            source=self.source,
            feature_names=self.pipeline.feature_names,
            training_meta=self.training_meta,
        )


class DetectorRegistry:
    """Holds every loaded detector; built once at startup, never per request."""

    def __init__(self, detectors: dict[str, LoadedDetector], default: str) -> None:
        if default not in detectors:
            raise ValueError(f"default detector {default!r} not among {sorted(detectors)}")
        self._detectors = dict(detectors)
        self.default_name = default

    def names(self) -> tuple[str, ...]:
        return tuple(self._detectors)

    def get(self, name: str) -> LoadedDetector:
        try:
            return self._detectors[name]
        except KeyError:
            raise KeyError(
                f"unknown detector {name!r}; available: {', '.join(sorted(self._detectors))}"
            ) from None

    def infos(self) -> tuple[DetectorInfo, ...]:
        return tuple(d.info() for d in self._detectors.values())

    # ─── Construction ────────────────────────────────────────────────────────

    @classmethod
    def demo(
        cls,
        *,
        scorer_spec: str = "stub",
        sample_path: Path | None = None,
        neural_checkpoint: str | None = None,
    ) -> DetectorRegistry:
        """Train demo classical detectors on the bundled sample corpus."""
        scorer = make_scorer(scorer_spec)
        pipeline = build_default_pipeline(scorer=scorer)
        records = assign_splits(read_jsonl(sample_path or _DEFAULT_SAMPLE), seed=0)
        source = "demo: trained at startup on the bundled synthetic sample corpus"
        detectors = {
            "classical": LoadedDetector(
                name="classical",
                kind="classical",
                description=(
                    "Calibrated logistic regression over interpretable statistical "
                    "features; sentence-level sibling model provides localization."
                ),
                source=source,
                pipeline=pipeline,
                sentence_baseline=train_sentence_baseline(records, pipeline, dataset_name="sample"),
                doc_baseline=train_baseline(records, pipeline, dataset_name="sample"),
                training_meta={"dataset": "data/sample/documents.jsonl", "split": "train"},
            )
        }
        if neural_checkpoint:
            detectors["neural"] = _load_neural(neural_checkpoint, pipeline)
        return cls(detectors, default="classical")

    @classmethod
    def from_artifacts(
        cls,
        artifact_dir: Path,
        *,
        scorer_spec: str = "stub",
        neural_checkpoint: str | None = None,
    ) -> DetectorRegistry:
        """Load trained classical artifacts from ``artifact_dir``."""
        scorer = make_scorer(scorer_spec)
        pipeline = build_default_pipeline(scorer=scorer)
        doc = BaselineDetector.load(artifact_dir / "doc_baseline.json")
        sent = BaselineDetector.load(artifact_dir / "sentence_baseline.json")
        for model, label in ((doc, "doc_baseline"), (sent, "sentence_baseline")):
            if model.feature_names != pipeline.feature_names:
                raise ValueError(
                    f"{label} feature schema does not match the configured pipeline; "
                    f"was it trained with a different scorer setting?"
                )
        detectors = {
            "classical": LoadedDetector(
                name="classical",
                kind="classical",
                description="Calibrated logistic regression loaded from trained artifacts.",
                source=str(artifact_dir),
                pipeline=pipeline,
                sentence_baseline=sent,
                doc_baseline=doc,
                training_meta={str(k): str(v) for k, v in doc.training_meta.items()},
            )
        }
        if neural_checkpoint:
            detectors["neural"] = _load_neural(neural_checkpoint, pipeline)
        return cls(detectors, default="classical")

    @classmethod
    def from_env(cls) -> DetectorRegistry:
        scorer_spec = os.environ.get("ORIGIN_SCORER", "stub")
        neural_checkpoint = os.environ.get("ORIGIN_NEURAL_CHECKPOINT") or None
        artifact_dir = os.environ.get("ORIGIN_ARTIFACT_DIR")
        if artifact_dir:
            return cls.from_artifacts(
                Path(artifact_dir), scorer_spec=scorer_spec, neural_checkpoint=neural_checkpoint
            )
        sample = os.environ.get("ORIGIN_SAMPLE_DATASET")
        return cls.demo(
            scorer_spec=scorer_spec,
            sample_path=Path(sample) if sample else None,
            neural_checkpoint=neural_checkpoint,
        )


def _load_neural(checkpoint: str, pipeline: FeaturePipeline) -> LoadedDetector:
    detector = NeuralDetector.from_checkpoint(checkpoint)
    return LoadedDetector(
        name="neural",
        kind="neural",
        description="Sentence-granular transformer classifier with document aggregation.",
        source=checkpoint,
        pipeline=pipeline,
        neural=detector,
        training_meta={"checkpoint": checkpoint},
    )
