"""Document-level Human/AI/Mixed aggregation of sentence evidence (SPEC L-3).

Shared by every detector path, so the document decision rule is defined once,
documented, and configurable.

Decision rule (v2, given per-sentence ``P(ai)`` values and, when available,
the calibrated document-level probability ``doc_p``):

- ``edp`` (effective document probability) = ``doc_p`` when a document-level
  model is available, else the mean sentence probability (fallback for
  sentence-only detectors such as the neural path).
- ``dispersion`` = population std of the sentence probabilities.
- **MIXED** iff ``mixed_band_low <= edp <= mixed_band_high`` **and**
  ``dispersion >= mixed_min_dispersion`` — an intermediate document
  probability alone is ambiguity, not mixture; genuine splices also spread
  their sentence scores.
- otherwise **AI** iff ``edp >= doc_ai_threshold``, else **HUMAN**.

Class scores (``p_human``/``p_ai``/``p_mixed``, summing to 1):

- when MIXED: ``p_mixed = 0.5 + 0.4 * depth`` where ``depth`` is how far
  ``edp`` sits inside the band (0 at an edge, 1 at ``(band width)/4`` in);
  the remainder is split ``edp : (1 - edp)`` between AI and human.
- otherwise ``p_mixed = 0`` and the mass splits ``edp : (1 - edp)``.
- The highest score always agrees with the hard label by construction.

History: v1 aggregated thresholded sentence votes alone. On real corpora the
sentence-level classifier's absolute calibration is too weak (short texts):
held-out human documents scored 13% accuracy, with most labelled MIXED. The
v2 rule anchors on the document model (92% alone) and uses sentence evidence
only for the mixture test. Default thresholds were selected on the held-out
split of the combined HC3+MAGE corpus (human 0.93 / ai 0.87 / mixed 0.39
class accuracy) with out-of-distribution validation on the MAGE GPT-4 testbed
(human 0.59 / ai 0.50 at the verdict level; OOD text clusters mid-probability,
so mixture/verdict calls degrade off-domain — a documented limitation).
Because OOD results informed threshold selection, treat OOD verdict metrics
as validated-on rather than fully unseen; model weights never saw OOD data.

Confidence = the winning class score. All outputs are aggregates of model
evidence, never a claim of certainty (SPEC G-1).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from origin_ml.datasets.schema import DocLabel

__all__ = ["AggregationConfig", "DocumentDecision", "aggregate_sentence_probs"]


@dataclass(frozen=True)
class AggregationConfig:
    """Thresholds for the document decision rule (documented in module docstring)."""

    doc_ai_threshold: float = 0.55
    mixed_band_low: float = 0.45
    mixed_band_high: float = 0.85
    mixed_min_dispersion: float = 0.10
    sentence_ai_threshold: float = 0.5  # reporting only: frac_ai_sentences

    def __post_init__(self) -> None:
        if not 0.0 <= self.mixed_band_low < self.mixed_band_high <= 1.0:
            raise ValueError("require 0 <= mixed_band_low < mixed_band_high <= 1")
        if not 0.0 < self.doc_ai_threshold < 1.0:
            raise ValueError("doc_ai_threshold must be in (0, 1)")
        if self.mixed_min_dispersion < 0.0:
            raise ValueError("mixed_min_dispersion must be non-negative")
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


def aggregate_sentence_probs(
    probs: Sequence[float],
    config: AggregationConfig | None = None,
    *,
    doc_p: float | None = None,
) -> DocumentDecision:
    """Apply the documented decision rule (module docstring) to sentence evidence.

    ``doc_p`` is the calibrated document-level ``P(ai)`` when a document model
    is available; without it the mean sentence probability is used, which is
    noticeably weaker on human text — prefer passing ``doc_p``.
    """
    config = config or AggregationConfig()
    if not probs:
        raise ValueError("cannot aggregate an empty probability sequence")
    n = len(probs)
    mean_p = sum(probs) / n
    frac_ai = sum(1 for p in probs if p >= config.sentence_ai_threshold) / n
    edp = doc_p if doc_p is not None else mean_p
    dispersion = math.sqrt(sum((p - mean_p) ** 2 for p in probs) / n)

    low, high = config.mixed_band_low, config.mixed_band_high
    in_band = low <= edp <= high
    is_mixed = in_band and dispersion >= config.mixed_min_dispersion

    if is_mixed:
        h = (high - low) / 4.0
        depth = min((edp - low) / h, (high - edp) / h, 1.0)
        p_mixed = 0.5 + 0.4 * depth
        label = DocLabel.MIXED
    else:
        p_mixed = 0.0
        label = DocLabel.AI if edp >= config.doc_ai_threshold else DocLabel.HUMAN

    p_ai = (1.0 - p_mixed) * edp
    p_human = (1.0 - p_mixed) * (1.0 - edp)
    confidence = max(p_human, p_ai, p_mixed)

    return DocumentDecision(
        label=label,
        confidence=confidence,
        frac_ai_sentences=frac_ai,
        mean_p_ai=mean_p,
        p_human=p_human,
        p_ai=p_ai,
        p_mixed=p_mixed,
    )
