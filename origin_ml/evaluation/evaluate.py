"""Record-level evaluation over detector systems and dataset slices (SPEC E-2, E-3).

An :class:`AblationSystem` abstracts any detector configuration behind two
callables — document probability and per-sentence probabilities — so the same
evaluation code measures classical, neural, and combined systems.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from origin_ml.datasets.schema import DocLabel, DocumentRecord, SegmentLabel
from origin_ml.detectors.aggregation import AggregationConfig, aggregate_sentence_probs
from origin_ml.evaluation.metrics import ClassificationMetrics, compute_binary_metrics
from origin_ml.text.segmentation import SentenceSpan, segment_sentences

__all__ = [
    "AblationSystem",
    "MixedDocSummary",
    "evaluate_doc_classification",
    "evaluate_localization",
    "evaluate_mixed_doc_labels",
    "sentence_truth",
]

SentenceScores = list[tuple[SentenceSpan, float]]


@dataclass(frozen=True)
class AblationSystem:
    """A named detector configuration under evaluation (SPEC E-4)."""

    name: str
    description: str
    doc_p_ai: Callable[[str], float]
    sentence_p_ai: Callable[[str], SentenceScores]


class MixedDocSummary(BaseModel):
    """Document-label behaviour on mixed documents (SPEC E-3)."""

    model_config = ConfigDict(frozen=True)

    n: int
    frac_labelled_mixed: float


def evaluate_doc_classification(
    records: Sequence[DocumentRecord], system: AblationSystem, *, threshold: float = 0.5
) -> ClassificationMetrics:
    """Binary human-vs-ai metrics over the pure documents of ``records``."""
    pure = [r for r in records if r.label in (DocLabel.HUMAN, DocLabel.AI)]
    if not pure:
        raise ValueError("no pure human/ai records to evaluate")
    y = [1 if r.label is DocLabel.AI else 0 for r in pure]
    p = [system.doc_p_ai(r.text) for r in pure]
    return compute_binary_metrics(y, p, threshold=threshold)


def sentence_truth(record: DocumentRecord, spans: Sequence[SentenceSpan]) -> list[int]:
    """Ground-truth sentence labels from a record (span midpoint rule)."""
    labels: list[int] = []
    for span in spans:
        if record.label is DocLabel.HUMAN:
            labels.append(0)
        elif record.label is DocLabel.AI:
            labels.append(1)
        else:
            midpoint = (span.start + span.end) / 2
            labels.append(
                int(
                    any(
                        s.label is SegmentLabel.AI and s.start <= midpoint < s.end
                        for s in record.spans
                    )
                )
            )
    return labels


def evaluate_localization(
    records: Sequence[DocumentRecord], system: AblationSystem, *, threshold: float = 0.5
) -> ClassificationMetrics:
    """Sentence-level localization metrics over mixed documents (SPEC E-2)."""
    mixed = [r for r in records if r.label is DocLabel.MIXED]
    if not mixed:
        raise ValueError("no mixed records to evaluate localization on")
    y_all: list[int] = []
    p_all: list[float] = []
    for record in mixed:
        scored = system.sentence_p_ai(record.text)
        spans = [span for span, _ in scored]
        y_all.extend(sentence_truth(record, spans))
        p_all.extend(p for _, p in scored)
    return compute_binary_metrics(y_all, p_all, threshold=threshold)


def evaluate_mixed_doc_labels(
    records: Sequence[DocumentRecord],
    system: AblationSystem,
    *,
    aggregation: AggregationConfig | None = None,
) -> MixedDocSummary:
    """How often mixed documents receive the MIXED label under aggregation."""
    mixed = [r for r in records if r.label is DocLabel.MIXED]
    if not mixed:
        raise ValueError("no mixed records to evaluate")
    hits = 0
    for record in mixed:
        probs = [p for _, p in system.sentence_p_ai(record.text)]
        decision = aggregate_sentence_probs(probs, aggregation)
        hits += int(decision.label is DocLabel.MIXED)
    return MixedDocSummary(n=len(mixed), frac_labelled_mixed=hits / len(mixed))


def default_sentence_scorer(
    probs_fn: Callable[[list[str]], list[float]],
) -> Callable[[str], SentenceScores]:
    """Adapt a batch sentence-probability function to the system interface."""

    def score(text: str) -> SentenceScores:
        spans = segment_sentences(text)
        if not spans:
            return []
        probs = probs_fn([span.text for span in spans])
        return list(zip(spans, probs, strict=True))

    return score
