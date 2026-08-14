"""Feature pipeline composing extractors into one named vector (SPEC F-12)."""

from __future__ import annotations

from collections.abc import Sequence

from origin_ml.features.base import AnalyzedText, FeatureExtractor, FeatureVector
from origin_ml.features.lexical import LexicalDiversityExtractor
from origin_ml.features.perplexity import PerplexityExtractor
from origin_ml.features.repetition import RepetitionExtractor
from origin_ml.features.sentences import SentenceStatsExtractor
from origin_ml.features.style import StyleExtractor
from origin_ml.scoring.base import Scorer

__all__ = ["FeaturePipeline", "build_default_pipeline"]


class FeaturePipeline:
    """Runs a fixed sequence of extractors over a document.

    The output feature order is the concatenation of each extractor's
    ``feature_names`` in construction order, giving detectors a stable,
    serializable feature schema.
    """

    def __init__(self, extractors: Sequence[FeatureExtractor], scorer: Scorer | None = None):
        if not extractors:
            raise ValueError("FeaturePipeline needs at least one extractor")
        names: list[str] = []
        for extractor in extractors:
            names.extend(extractor.feature_names)
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise ValueError(f"duplicate feature names across extractors: {sorted(duplicates)}")
        missing_scorer = [
            type(e).__name__ for e in extractors if e.requires_scoring and scorer is None
        ]
        if missing_scorer:
            raise ValueError(f"extractors require a scorer but none was given: {missing_scorer}")
        self._extractors = tuple(extractors)
        self._scorer = scorer
        self._names = tuple(names)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._names

    @property
    def scorer(self) -> Scorer | None:
        return self._scorer

    @property
    def extractors(self) -> tuple[FeatureExtractor, ...]:
        return self._extractors

    def analyze(self, text: str) -> AnalyzedText:
        return AnalyzedText.analyze(text, self._scorer)

    def extract(self, text: str) -> FeatureVector:
        return self.extract_from(self.analyze(text))

    def extract_from(self, doc: AnalyzedText) -> FeatureVector:
        """Extract from a pre-analyzed document (lets callers reuse scoring)."""
        values: list[float] = []
        for extractor in self._extractors:
            result = extractor.extract(doc)
            if set(result) != set(extractor.feature_names):
                raise RuntimeError(
                    f"{type(extractor).__name__} returned features "
                    f"{sorted(result)} != declared {sorted(extractor.feature_names)}"
                )
            values.extend(result[name] for name in extractor.feature_names)
        return FeatureVector(names=self._names, values=tuple(values))


def build_default_pipeline(scorer: Scorer | None = None) -> FeaturePipeline:
    """The canonical Origin feature set.

    With a scorer: perplexity/surprisal/entropy features plus all statistical
    features. Without one: statistical features only (used where no LM is
    available).
    """
    extractors: list[FeatureExtractor] = []
    if scorer is not None:
        extractors.append(PerplexityExtractor())
    extractors.extend(
        [
            SentenceStatsExtractor(),
            LexicalDiversityExtractor(),
            RepetitionExtractor(),
            StyleExtractor(),
        ]
    )
    return FeaturePipeline(extractors, scorer=scorer)
