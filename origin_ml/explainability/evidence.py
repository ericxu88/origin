"""Evidence bundle models (SPEC X-1..X-4).

Every evidence section carries an explicit ``kind`` tag distinguishing
**heuristic** evidence (statistical measurements: surprisal, perplexity,
feature values) from **model** evidence (learned-classifier probabilities) —
SPEC G-2/X-4. These are pydantic models so the API can serve them verbatim
and the frontend renders from one schema.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "DistributionComparisonSection",
    "DocumentFeatureSummary",
    "EvidenceBundle",
    "FeatureComparison",
    "SentenceHeat",
    "SentenceStatistics",
    "TokenSurprisal",
    "TokenSurprisalSeries",
]


class TokenSurprisal(BaseModel):
    """One token's surprisal, with exact document offsets (SPEC X-2)."""

    model_config = ConfigDict(frozen=True)

    text: str
    start: int = Field(ge=0)
    end: int
    logprob: float
    surprisal_bits: float
    entropy: float | None = None


class TokenSurprisalSeries(BaseModel):
    """Token-level surprisal visualization data (heuristic evidence)."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["heuristic"] = "heuristic"
    scorer: str
    tokens: tuple[TokenSurprisal, ...]


class SentenceHeat(BaseModel):
    """Learned per-sentence AI probability for the heatmap (model evidence)."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["model"] = "model"
    text: str
    start: int = Field(ge=0)
    end: int
    p_ai: float = Field(ge=0, le=1)


class SentenceStatistics(BaseModel):
    """Per-sentence statistical measurements (heuristic evidence, SPEC L-2)."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["heuristic"] = "heuristic"
    start: int = Field(ge=0)
    end: int
    n_words: int
    n_scored_tokens: int
    perplexity: float | None = None
    mean_surprisal_bits: float | None = None
    max_surprisal_bits: float | None = None


class DocumentFeatureSummary(BaseModel):
    """The full document feature vector (heuristic evidence, SPEC X-1)."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["heuristic"] = "heuristic"
    scorer: str | None
    features: dict[str, float]


class FeatureComparison(BaseModel):
    """Observed feature value vs. training-time class distributions (SPEC X-3)."""

    model_config = ConfigDict(frozen=True)

    feature: str
    observed: float
    human_mean: float
    human_std: float
    ai_mean: float
    ai_std: float
    z_vs_human: float
    z_vs_ai: float
    closer_to: Literal["human", "ai", "similar"]


class DistributionComparisonSection(BaseModel):
    """Observed-vs-expected comparison, present only when the loaded artifact
    embeds training distributions (heuristic evidence)."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["heuristic"] = "heuristic"
    comparisons: tuple[FeatureComparison, ...]


class EvidenceBundle(BaseModel):
    """Everything the UI needs to show *why*, not just a probability."""

    model_config = ConfigDict(frozen=True)

    heatmap: tuple[SentenceHeat, ...]
    sentence_statistics: tuple[SentenceStatistics, ...]
    token_surprisals: TokenSurprisalSeries | None
    document_features: DocumentFeatureSummary
    distribution_comparison: DistributionComparisonSection | None
