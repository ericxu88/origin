"""Vocabulary-frequency and style features (SPEC F-11).

Rates are per word token unless stated otherwise:

- ``style.stopword_ratio`` — fraction of word tokens in :data:`STOPWORDS`.
- ``style.common_word_ratio`` — fraction in ``STOPWORDS | COMMON_WORDS``
  (the high-frequency vocabulary band).
- ``style.avg_word_length`` — mean characters per word token.
- ``style.punct_char_ratio`` — punctuation characters / non-whitespace chars.
- ``style.comma_rate`` / ``style.semicolon_rate`` — marks per word token.
- ``style.digit_char_ratio`` — digit characters / non-whitespace characters.
- ``style.allcaps_word_ratio`` — fully-uppercase words of length > 1.
"""

from __future__ import annotations

import string

from origin_ml.features.base import AnalyzedText
from origin_ml.features.wordlists import COMMON_WORDS, STOPWORDS

__all__ = ["StyleExtractor"]

_PUNCT = set(string.punctuation)
_HIGH_FREQUENCY_BAND = STOPWORDS | COMMON_WORDS


class StyleExtractor:
    """Vocabulary-band and punctuation style statistics."""

    _NAMES = (
        "style.stopword_ratio",
        "style.common_word_ratio",
        "style.avg_word_length",
        "style.punct_char_ratio",
        "style.comma_rate",
        "style.semicolon_rate",
        "style.digit_char_ratio",
        "style.allcaps_word_ratio",
    )

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._NAMES

    @property
    def requires_scoring(self) -> bool:
        return False

    def extract(self, doc: AnalyzedText) -> dict[str, float]:
        words = [w.text for w in doc.words]
        lowered = [w.lower() for w in words]
        n_words = len(words)

        non_space = [c for c in doc.text if not c.isspace()]
        n_chars = len(non_space)

        def per_word(count: int) -> float:
            return count / n_words if n_words else 0.0

        def per_char(count: int) -> float:
            return count / n_chars if n_chars else 0.0

        return {
            "style.stopword_ratio": per_word(sum(1 for w in lowered if w in STOPWORDS)),
            "style.common_word_ratio": per_word(
                sum(1 for w in lowered if w in _HIGH_FREQUENCY_BAND)
            ),
            "style.avg_word_length": (sum(len(w) for w in words) / n_words) if n_words else 0.0,
            "style.punct_char_ratio": per_char(sum(1 for c in non_space if c in _PUNCT)),
            "style.comma_rate": per_word(doc.text.count(",")),
            "style.semicolon_rate": per_word(doc.text.count(";")),
            "style.digit_char_ratio": per_char(sum(1 for c in non_space if c.isdigit())),
            "style.allcaps_word_ratio": per_word(
                sum(1 for w in words if len(w) > 1 and w.isupper())
            ),
        }
