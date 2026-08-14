"""Interpretable statistical feature engine (SPEC §3.1)."""

from origin_ml.features.base import AnalyzedText, FeatureExtractor, FeatureVector
from origin_ml.features.lexical import LexicalDiversityExtractor
from origin_ml.features.perplexity import PerplexityExtractor
from origin_ml.features.pipeline import FeaturePipeline, build_default_pipeline
from origin_ml.features.repetition import RepetitionExtractor
from origin_ml.features.sentences import SentenceStatsExtractor
from origin_ml.features.style import StyleExtractor

__all__ = [
    "AnalyzedText",
    "FeatureExtractor",
    "FeaturePipeline",
    "FeatureVector",
    "LexicalDiversityExtractor",
    "PerplexityExtractor",
    "RepetitionExtractor",
    "SentenceStatsExtractor",
    "StyleExtractor",
    "build_default_pipeline",
]
