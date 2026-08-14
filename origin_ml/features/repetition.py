"""Repetition features (SPEC F-10).

- ``rep.distinct_n`` — distinct n-gram ratio ``|unique n-grams| / |n-grams|``
  over lowercased words (1.0 when the document has no n-grams of that order).
  Lower values indicate more repetition; degenerate sampled text scores low.
- ``rep.top_bigram_share`` / ``rep.top_trigram_share`` — occurrence share of
  the single most frequent n-gram (0.0 when there are none).
- ``rep.repeated_sentence_start_ratio`` — ``1 - |unique first words| / |sentences|``;
  captures formulaic paragraph patterns ("Additionally, ... Additionally, ...").
"""

from __future__ import annotations

from collections import Counter

from origin_ml.features.base import AnalyzedText

__all__ = ["RepetitionExtractor"]


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


class RepetitionExtractor:
    """Distinct-n and top-n-gram-share repetition statistics."""

    _NAMES = (
        "rep.distinct_1",
        "rep.distinct_2",
        "rep.distinct_3",
        "rep.top_bigram_share",
        "rep.top_trigram_share",
        "rep.repeated_sentence_start_ratio",
    )

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._NAMES

    @property
    def requires_scoring(self) -> bool:
        return False

    def extract(self, doc: AnalyzedText) -> dict[str, float]:
        tokens = [w.text.lower() for w in doc.words]

        def distinct(n: int) -> float:
            grams = _ngrams(tokens, n)
            return len(set(grams)) / len(grams) if grams else 1.0

        def top_share(n: int) -> float:
            grams = _ngrams(tokens, n)
            if not grams:
                return 0.0
            ((_, top_count),) = Counter(grams).most_common(1)
            return top_count / len(grams)

        first_words = [
            next(
                (w.text.lower() for w in doc.words if s.start <= w.start < s.end),
                "",
            )
            for s in doc.sentences
        ]
        first_words = [w for w in first_words if w]
        repeated_starts = 1.0 - len(set(first_words)) / len(first_words) if first_words else 0.0

        return {
            "rep.distinct_1": distinct(1),
            "rep.distinct_2": distinct(2),
            "rep.distinct_3": distinct(3),
            "rep.top_bigram_share": top_share(2),
            "rep.top_trigram_share": top_share(3),
            "rep.repeated_sentence_start_ratio": repeated_starts,
        }
