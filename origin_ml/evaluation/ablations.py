"""Ablation systems: which signals drive detection (SPEC E-4, RQ6).

The five canonical configurations compare signal families:

- ``perplexity_only``     — classical LR on LM features (``ppl.*``) alone.
- ``statistical_features``— classical LR on non-LM statistical features only.
- ``classical_full``      — classical LR on the full feature set.
- ``neural``              — the sentence-granular transformer detector.
- ``combined``            — mean of ``classical_full`` and ``neural`` probabilities.

Each system trains only on the provided training records and is evaluated by
:mod:`origin_ml.evaluation.evaluate` on held-out slices.
"""

from __future__ import annotations

import numpy as np

from origin_ml.datasets.schema import DocumentRecord
from origin_ml.evaluation.evaluate import AblationSystem, SentenceScores, default_sentence_scorer
from origin_ml.features.lexical import LexicalDiversityExtractor
from origin_ml.features.perplexity import PerplexityExtractor
from origin_ml.features.pipeline import FeaturePipeline, build_default_pipeline
from origin_ml.features.repetition import RepetitionExtractor
from origin_ml.features.sentences import SentenceStatsExtractor
from origin_ml.features.style import StyleExtractor
from origin_ml.scoring.base import Scorer
from origin_ml.training.classical import train_baseline, train_sentence_baseline
from origin_ml.training.neural import NeuralTrainConfig, train_neural

__all__ = ["ABLATION_NAMES", "build_ablation_system"]

ABLATION_NAMES = (
    "perplexity_only",
    "statistical_features",
    "classical_full",
    "neural",
    "combined",
)


def _classical_system(
    name: str,
    description: str,
    pipeline: FeaturePipeline,
    train_records: list[DocumentRecord],
    seed: int,
) -> AblationSystem:
    from origin_ml.detectors.classical import BaselineConfig

    config = BaselineConfig(seed=seed)
    doc_model = train_baseline(train_records, pipeline, config=config)
    sent_model = train_sentence_baseline(train_records, pipeline, config=config)

    def doc_p(text: str) -> float:
        vector = pipeline.extract(text)
        return doc_model.predict_proba_one(np.asarray(vector.values, dtype=np.float64))

    def batch_sentence_probs(sentences: list[str]) -> list[float]:
        matrix = np.asarray([pipeline.extract(s).values for s in sentences], dtype=np.float64)
        return [float(p) for p in sent_model.predict_proba(matrix)]

    return AblationSystem(
        name=name,
        description=description,
        doc_p_ai=doc_p,
        sentence_p_ai=default_sentence_scorer(batch_sentence_probs),
    )


def build_ablation_system(
    name: str,
    train_records: list[DocumentRecord],
    *,
    scorer: Scorer | None,
    neural_checkpoint: str,
    seed: int = 0,
    neural_epochs: int = 3,
) -> AblationSystem:
    """Construct and train one named ablation system (SPEC E-4)."""
    if name == "perplexity_only":
        if scorer is None:
            raise ValueError("perplexity_only requires a scorer")
        pipeline = FeaturePipeline([PerplexityExtractor()], scorer=scorer)
        return _classical_system(
            name,
            "classical LR on LM perplexity/surprisal features only",
            pipeline,
            train_records,
            seed,
        )
    if name == "statistical_features":
        pipeline = FeaturePipeline(
            [
                SentenceStatsExtractor(),
                LexicalDiversityExtractor(),
                RepetitionExtractor(),
                StyleExtractor(),
            ]
        )
        return _classical_system(
            name,
            "classical LR on non-LM statistical features only",
            pipeline,
            train_records,
            seed,
        )
    if name == "classical_full":
        pipeline = build_default_pipeline(scorer=scorer)
        return _classical_system(
            name, "classical LR on the full feature set", pipeline, train_records, seed
        )
    if name == "neural":
        detector, _ = train_neural(
            train_records,
            NeuralTrainConfig(
                checkpoint=neural_checkpoint, epochs=neural_epochs, seed=seed, device="cpu"
            ),
        )

        def neural_doc_p(text: str) -> float:
            return detector.predict_document(text).mean_p_ai

        return AblationSystem(
            name=name,
            description=f"sentence-granular transformer ({neural_checkpoint})",
            doc_p_ai=neural_doc_p,
            sentence_p_ai=default_sentence_scorer(detector.sentence_probs),
        )
    if name == "combined":
        classical = build_ablation_system(
            "classical_full",
            train_records,
            scorer=scorer,
            neural_checkpoint=neural_checkpoint,
            seed=seed,
            neural_epochs=neural_epochs,
        )
        neural = build_ablation_system(
            "neural",
            train_records,
            scorer=scorer,
            neural_checkpoint=neural_checkpoint,
            seed=seed,
            neural_epochs=neural_epochs,
        )

        def combined_doc(text: str) -> float:
            return (classical.doc_p_ai(text) + neural.doc_p_ai(text)) / 2

        def combined_sentences(text: str) -> SentenceScores:
            a = classical.sentence_p_ai(text)
            b = neural.sentence_p_ai(text)
            return [(span_a, (p_a + p_b) / 2) for (span_a, p_a), (_, p_b) in zip(a, b, strict=True)]

        return AblationSystem(
            name=name,
            description="mean of classical_full and neural probabilities",
            doc_p_ai=combined_doc,
            sentence_p_ai=combined_sentences,
        )
    raise ValueError(f"unknown ablation {name!r}; expected one of {ABLATION_NAMES}")
