"""Explainability: evidence assembly and end-to-end analysis (SPEC §3.5)."""

from origin_ml.explainability.analyze import DISCLAIMER, AnalysisResult, analyze_document
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

__all__ = [
    "DISCLAIMER",
    "AnalysisResult",
    "DistributionComparisonSection",
    "DocumentFeatureSummary",
    "EvidenceBundle",
    "FeatureComparison",
    "SentenceHeat",
    "SentenceStatistics",
    "TokenSurprisal",
    "TokenSurprisalSeries",
    "analyze_document",
]
