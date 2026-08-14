"""End-to-end document analysis: localization + evidence assembly (SPEC L-2, X-1..X-4).

:func:`analyze_document` is the single entry point behind the API, CLI, and
tests. It composes:

1. sentence segmentation with exact offsets (L-1),
2. per-sentence AI probabilities from the neural detector when provided,
   otherwise from a **sentence-level** classical baseline (document-level
   models must never be applied per sentence — sentence feature vectors lie
   outside their training distribution),
3. the shared, documented Human/AI/Mixed aggregation rule (L-3),
4. the tagged evidence bundle (heuristic vs model — G-2/X-4), including
   observed-vs-expected feature comparisons when the baseline artifact embeds
   training distributions (X-3).

Detection output is always framed as probabilistic evidence (G-1): every
result carries the standard disclaimer.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from origin_ml.datasets.schema import DocLabel
from origin_ml.detectors.aggregation import AggregationConfig, aggregate_sentence_probs
from origin_ml.detectors.classical import BaselineDetector
from origin_ml.detectors.neural import NeuralDetector
from origin_ml.explainability.evidence import (
    DistributionComparisonSection,
    DocumentFeatureSummary,
    EvidenceBundle,
    FeatureComparison,
    SentenceHeat,
    SentenceStatistics,
    TokenSurprisal,
    TokenSurprisalSeries,
)
from origin_ml.features.base import AnalyzedText
from origin_ml.features.pipeline import FeaturePipeline
from origin_ml.scoring.base import perplexity

__all__ = ["DISCLAIMER", "AnalysisResult", "ClassProbabilities", "analyze_document"]

DISCLAIMER = (
    "Origin reports statistical evidence, not proof. AI-text detection is "
    "probabilistic and can be wrong — especially for short, heavily edited, "
    "or non-native-English writing. Treat results as a signal for review, "
    "never as ground truth."
)

_Z_SIMILAR_MARGIN = 0.25  # |z| difference below this counts as "similar"


class ClassProbabilities(BaseModel):
    """Soft Human/AI/Mixed class scores (sum to 1; argmax matches the label).

    Derived from the same statistic and thresholds as the decision rule — see
    :mod:`origin_ml.detectors.aggregation`. Model evidence, not certainty.
    """

    model_config = ConfigDict(frozen=True)

    human: float = Field(ge=0, le=1)
    ai: float = Field(ge=0, le=1)
    mixed: float = Field(ge=0, le=1)


class AnalysisResult(BaseModel):
    """Structured document + sentence prediction with evidence (SPEC L-2)."""

    model_config = ConfigDict(frozen=True)

    label: DocLabel
    confidence: float = Field(ge=0, le=1)
    class_probabilities: ClassProbabilities
    mean_p_ai: float = Field(ge=0, le=1)
    frac_ai_sentences: float = Field(ge=0, le=1)
    document_p_ai: float | None = None
    detector: str
    disclaimer: str = DISCLAIMER
    evidence: EvidenceBundle


def _sentence_probs_classical(
    doc: AnalyzedText, pipeline: FeaturePipeline, sentence_baseline: BaselineDetector
) -> list[float]:
    matrix = np.asarray(
        [pipeline.extract(span.text).values for span in doc.sentences], dtype=np.float64
    )
    return [float(p) for p in sentence_baseline.predict_proba(matrix)]


def _sentence_statistics(doc: AnalyzedText) -> tuple[SentenceStatistics, ...]:
    stats: list[SentenceStatistics] = []
    for span in doc.sentences:
        n_words = sum(1 for w in doc.words if span.start <= w.start < span.end)
        scored = doc.scored
        tokens = scored.tokens_in_span(span.start, span.end) if scored is not None else []
        surprisals = [t.surprisal_bits for t in tokens]
        stats.append(
            SentenceStatistics(
                start=span.start,
                end=span.end,
                n_words=n_words,
                n_scored_tokens=len(tokens),
                perplexity=perplexity([t.logprob for t in tokens]) if tokens else None,
                mean_surprisal_bits=(sum(surprisals) / len(surprisals)) if surprisals else None,
                max_surprisal_bits=max(surprisals) if surprisals else None,
            )
        )
    return tuple(stats)


def _distribution_comparison(
    features: dict[str, float], baseline: BaselineDetector
) -> DistributionComparisonSection | None:
    distributions = baseline.feature_distributions
    if not distributions.get("human") or not distributions.get("ai"):
        return None
    comparisons: list[FeatureComparison] = []
    for name, observed in features.items():
        human = distributions["human"].get(name)
        ai = distributions["ai"].get(name)
        if human is None or ai is None:
            continue
        z_human = (observed - human.mean) / human.std if human.std > 0 else 0.0
        z_ai = (observed - ai.mean) / ai.std if ai.std > 0 else 0.0
        if abs(abs(z_human) - abs(z_ai)) < _Z_SIMILAR_MARGIN:
            closer: str = "similar"
        else:
            closer = "human" if abs(z_human) < abs(z_ai) else "ai"
        comparisons.append(
            FeatureComparison(
                feature=name,
                observed=observed,
                human_mean=human.mean,
                human_std=human.std,
                ai_mean=ai.mean,
                ai_std=ai.std,
                z_vs_human=z_human,
                z_vs_ai=z_ai,
                closer_to=closer,  # type: ignore[arg-type]
            )
        )
    return DistributionComparisonSection(comparisons=tuple(comparisons))


def analyze_document(
    text: str,
    *,
    pipeline: FeaturePipeline,
    sentence_baseline: BaselineDetector | None = None,
    doc_baseline: BaselineDetector | None = None,
    neural: NeuralDetector | None = None,
    aggregation: AggregationConfig | None = None,
) -> AnalysisResult:
    """Analyze ``text`` end to end.

    Requires a sentence-capable detector: ``neural`` (preferred when given) or
    ``sentence_baseline`` (a classical model trained on sentence examples via
    :func:`origin_ml.training.train_sentence_baseline`). ``doc_baseline``
    optionally adds the document-level probability and observed-vs-expected
    distribution comparisons (X-3).
    """
    if sentence_baseline is None and neural is None:
        raise ValueError(
            "analyze_document requires a sentence-capable detector: neural or sentence_baseline"
        )
    if not text.strip():
        raise ValueError("document is empty")

    doc = pipeline.analyze(text)
    if not doc.sentences:
        raise ValueError("document contains no sentences")
    doc_vector = pipeline.extract_from(doc)
    features = doc_vector.as_dict()

    if neural is not None:
        probs = neural.sentence_probs([span.text for span in doc.sentences])
        detector_name = f"neural({neural.checkpoint})"
    else:
        assert sentence_baseline is not None
        probs = _sentence_probs_classical(doc, pipeline, sentence_baseline)
        detector_name = "baseline-logreg"

    document_p_ai: float | None = None
    distribution_comparison: DistributionComparisonSection | None = None
    if doc_baseline is not None:
        document_p_ai = doc_baseline.predict_proba_one(
            np.asarray(doc_vector.values, dtype=np.float64)
        )
        distribution_comparison = _distribution_comparison(features, doc_baseline)

    decision = aggregate_sentence_probs(probs, aggregation, doc_p=document_p_ai)

    token_surprisals: TokenSurprisalSeries | None = None
    if doc.scored is not None and doc.scored.tokens:
        token_surprisals = TokenSurprisalSeries(
            scorer=doc.scored.scorer_name,
            tokens=tuple(
                TokenSurprisal(
                    text=t.text,
                    start=t.start,
                    end=t.end,
                    logprob=t.logprob,
                    surprisal_bits=t.surprisal_bits,
                    entropy=t.entropy,
                )
                for t in doc.scored.tokens
            ),
        )

    scorer = pipeline.scorer
    evidence = EvidenceBundle(
        heatmap=tuple(
            SentenceHeat(text=span.text, start=span.start, end=span.end, p_ai=p)
            for span, p in zip(doc.sentences, probs, strict=True)
        ),
        sentence_statistics=_sentence_statistics(doc),
        token_surprisals=token_surprisals,
        document_features=DocumentFeatureSummary(
            scorer=scorer.name if scorer is not None else None, features=features
        ),
        distribution_comparison=distribution_comparison,
    )
    return AnalysisResult(
        label=decision.label,
        confidence=decision.confidence,
        class_probabilities=ClassProbabilities(
            human=decision.p_human, ai=decision.p_ai, mixed=decision.p_mixed
        ),
        mean_p_ai=decision.mean_p_ai,
        frac_ai_sentences=decision.frac_ai_sentences,
        document_p_ai=document_p_ai,
        detector=detector_name,
        evidence=evidence,
    )
