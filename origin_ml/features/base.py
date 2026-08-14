"""Feature-engine abstractions (SPEC F-12).

A :class:`FeatureExtractor` computes a fixed, named set of float features from
an :class:`AnalyzedText` (text + sentence spans + word tokens + optional LM
scoring). A :class:`origin_ml.features.pipeline.FeaturePipeline` composes
extractors into a single named, ordered feature vector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from origin_ml.scoring.base import ScoredText, Scorer
from origin_ml.text.segmentation import SentenceSpan, segment_sentences
from origin_ml.text.words import WordSpan, tokenize_words

__all__ = ["AnalyzedText", "FeatureExtractor", "FeatureVector"]


@dataclass(frozen=True, slots=True)
class AnalyzedText:
    """A document prepared for feature extraction.

    Segmentation and scoring happen once here and are shared by all
    extractors; ``scored`` is ``None`` when no scorer is configured.
    """

    text: str
    sentences: tuple[SentenceSpan, ...]
    words: tuple[WordSpan, ...]
    scored: ScoredText | None = None

    @classmethod
    def analyze(cls, text: str, scorer: Scorer | None = None) -> AnalyzedText:
        return cls(
            text=text,
            sentences=tuple(segment_sentences(text)),
            words=tuple(tokenize_words(text)),
            scored=scorer.score(text) if scorer is not None else None,
        )


@runtime_checkable
class FeatureExtractor(Protocol):
    """Computes a fixed set of named float features from an AnalyzedText."""

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Names, in output order; globally unique across a pipeline."""
        ...

    @property
    def requires_scoring(self) -> bool:
        """Whether :meth:`extract` needs ``doc.scored`` to be present."""
        ...

    def extract(self, doc: AnalyzedText) -> dict[str, float]:
        """Return exactly the features declared in :attr:`feature_names`."""
        ...


@dataclass(frozen=True)
class FeatureVector:
    """An ordered, named feature vector."""

    names: tuple[str, ...] = field()
    values: tuple[float, ...] = field()

    def __post_init__(self) -> None:
        if len(self.names) != len(self.values):
            raise ValueError(
                f"names ({len(self.names)}) and values ({len(self.values)}) length mismatch"
            )

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.names, self.values, strict=True))

    def __getitem__(self, name: str) -> float:
        try:
            return self.values[self.names.index(name)]
        except ValueError:
            raise KeyError(name) from None
