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

Confidence is therefore always in ``[0, 1]`` and is an *aggregate of
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
    return DocumentDecision(
        label=label,
        confidence=confidence,
        frac_ai_sentences=frac_ai,
        mean_p_ai=mean_p,
    )
