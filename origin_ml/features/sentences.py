"""Sentence-length statistics and burstiness (SPEC F-6, F-8).

Human writing tends to vary sentence length more than sampled LLM text at low
temperature ("burstiness"); these features quantify that variation without any
language model.
"""

from __future__ import annotations

from origin_ml.features._stats import (
    burstiness,
    coeff_variation,
    safe_max,
    safe_mean,
    safe_min,
    safe_pstd,
)
from origin_ml.features.base import AnalyzedText

__all__ = ["SentenceStatsExtractor"]


class SentenceStatsExtractor:
    """Per-sentence length statistics in words and characters."""

    _NAMES = (
        "sent.count",
        "sent.mean_len_words",
        "sent.std_len_words",
        "sent.min_len_words",
        "sent.max_len_words",
        "sent.mean_len_chars",
        "sent.std_len_chars",
        "sent.len_words_cv",
        "sent.len_words_burstiness",
    )

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._NAMES

    @property
    def requires_scoring(self) -> bool:
        return False

    def extract(self, doc: AnalyzedText) -> dict[str, float]:
        word_lens: list[float] = []
        char_lens: list[float] = []
        for sentence in doc.sentences:
            n_words = sum(1 for w in doc.words if sentence.start <= w.start < sentence.end)
            word_lens.append(float(n_words))
            char_lens.append(float(sentence.end - sentence.start))

        return {
            "sent.count": float(len(doc.sentences)),
            "sent.mean_len_words": safe_mean(word_lens),
            "sent.std_len_words": safe_pstd(word_lens),
            "sent.min_len_words": safe_min(word_lens),
            "sent.max_len_words": safe_max(word_lens),
            "sent.mean_len_chars": safe_mean(char_lens),
            "sent.std_len_chars": safe_pstd(char_lens),
            "sent.len_words_cv": coeff_variation(word_lens),
            "sent.len_words_burstiness": burstiness(word_lens),
        }
