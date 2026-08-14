"""Detectors: classical baseline and neural (SPEC §3.2, §3.3)."""

from origin_ml.detectors.aggregation import (
    AggregationConfig,
    DocumentDecision,
    aggregate_sentence_probs,
)
from origin_ml.detectors.classical import (
    BaselineConfig,
    BaselineDetector,
    DistributionSummary,
    FeatureImportance,
)
from origin_ml.detectors.neural import NeuralDetector, NeuralPrediction, SentenceProbability

__all__ = [
    "AggregationConfig",
    "BaselineConfig",
    "BaselineDetector",
    "DistributionSummary",
    "DocumentDecision",
    "FeatureImportance",
    "NeuralDetector",
    "NeuralPrediction",
    "SentenceProbability",
    "aggregate_sentence_probs",
]
