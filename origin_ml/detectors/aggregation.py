"""Document-level Human/AI/Mixed aggregation of sentence evidence (SPEC L-3).

Shared by every detector path that produces per-sentence AI probabilities
(neural and classical-over-sentences), so the document decision rule is
defined once, documented, and configurable.

Decision rule (given per-sentence ``P(ai)`` values):

- ``frac_ai`` = fraction of sentences with ``P(ai) >= sentence_ai_threshold``.
- ``frac_ai >= mixed_high``  → **AI**;    confidence = mean ``P(ai)``.
- ``frac_ai <= mixed_low``   → **HUMAN**; confidence = mean ``1 - P(ai)``.
- otherwise                  → **MIXED**; confidence = mean ``2 * |P(ai) - 0.5|``
  (how decisively individual sentences vote, regardless of direction).

Class scores (``p_human``/``p_ai``/``p_mixed``, summing to 1) soften the same
rule: piecewise-linear class memberships over ``frac_ai`` whose transitions are
centred exactly on the two thresholds, with half-width
``h = min((mixed_high - mixed_low) / 4, mixed_low, 1 - mixed_high)``. Away from
a threshold the winning class scores 1.0; within ``±h`` of a threshold the two
adjacent classes share the mass linearly (50/50 exactly at the threshold). By
construction the highest score always agrees with the hard label, so the
percentages shown in the UI can never contradict the verdict.

Confidence and class scores are always in ``[0, 1]`` and are *aggregates of
per-sentence evidence*, never a claim of certainty (SPEC G-1).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from origin_ml.datasets.schema import DocLabel

__all__ = ["AggregationConfig", "DocumentDecision", "aggregate_sentence_probs"]


@dataclass(frozen=True)
class AggregationConfig:
    """Thresholds for the document decision rule (documented in module docstring)."""

    sentence_ai_threshold: float = 0.5
    mixed_low: float = 0.25
    mixed_high: float = 0.75

    def __post_init__(self) -> None:
        if not 0.0 <= self.mixed_low < self.mixed_high <= 1.0:
            raise ValueError("require 0 <= mixed_low < mixed_high <= 1")
        if not 0.0 < self.sentence_ai_threshold < 1.0:
            raise ValueError("sentence_ai_threshold must be in (0, 1)")


@dataclass(frozen=True)
class DocumentDecision:
    label: DocLabel
    confidence: float
    frac_ai_sentences: float
    mean_p_ai: float
    p_human: float
    p_ai: float
    p_mixed: float


def _class_scores(frac_ai: float, config: AggregationConfig) -> tuple[float, float, float]:
    """Soft (p_human, p_mixed, p_ai) memberships; see module docstring."""
    low, high = config.mixed_low, config.mixed_high
    h = min((high - low) / 4.0, low, 1.0 - high)
    if h <= 0.0:
        # Degenerate thresholds (mixed_low == 0 or mixed_high == 1): hard one-hot.
        if frac_ai >= high:
            return 0.0, 0.0, 1.0
        if frac_ai <= low:
            return 1.0, 0.0, 0.0
        return 0.0, 1.0, 0.0

    if frac_ai <= low - h:
        return 1.0, 0.0, 0.0
    if frac_ai < low + h:
        mixed = (frac_ai - (low - h)) / (2.0 * h)
        return 1.0 - mixed, mixed, 0.0
    if frac_ai <= high - h:
        return 0.0, 1.0, 0.0
    if frac_ai < high + h:
        ai = (frac_ai - (high - h)) / (2.0 * h)
        return 0.0, 1.0 - ai, ai
    return 0.0, 0.0, 1.0


def aggregate_sentence_probs(
    probs: Sequence[float], config: AggregationConfig | None = None
) -> DocumentDecision:
    """Apply the documented decision rule to per-sentence AI probabilities."""
    config = config or AggregationConfig()
    if not probs:
        raise ValueError("cannot aggregate an empty probability sequence")
    n = len(probs)
    mean_p = sum(probs) / n
    frac_ai = sum(1 for p in probs if p >= config.sentence_ai_threshold) / n

    if frac_ai >= config.mixed_high:
        label, confidence = DocLabel.AI, mean_p
    elif frac_ai <= config.mixed_low:
        label, confidence = DocLabel.HUMAN, 1.0 - mean_p
    else:
        label = DocLabel.MIXED
        confidence = sum(2.0 * abs(p - 0.5) for p in probs) / n

    p_human, p_mixed, p_ai = _class_scores(frac_ai, config)
    return DocumentDecision(
        label=label,
        confidence=confidence,
        frac_ai_sentences=frac_ai,
        mean_p_ai=mean_p,
        p_human=p_human,
        p_ai=p_ai,
        p_mixed=p_mixed,
    )
