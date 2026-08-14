"""Lexical diversity features (SPEC F-9).

- ``lex.ttr`` — type-token ratio ``|types| / |tokens|`` (lowercased words).
  TTR shrinks with document length, so it is complemented by:
- ``lex.mattr_w50`` — moving-average TTR: mean TTR over all sliding windows of
  50 tokens (Covington & McFall, 2010). Falls back to plain TTR for documents
  shorter than the window, where the two coincide in the limit.
- ``lex.hapax_ratio`` — fraction of types occurring exactly once.
"""

from __future__ import annotations

from collections import Counter

from origin_ml.features.base import AnalyzedText

__all__ = ["LexicalDiversityExtractor"]

_MATTR_WINDOW = 50


class LexicalDiversityExtractor:
    """Type-token ratio family over lowercased word tokens."""

    _NAMES = ("lex.ttr", "lex.mattr_w50", "lex.hapax_ratio")

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._NAMES

    @property
    def requires_scoring(self) -> bool:
        return False

    def extract(self, doc: AnalyzedText) -> dict[str, float]:
        tokens = [w.text.lower() for w in doc.words]
        if not tokens:
            return {"lex.ttr": 0.0, "lex.mattr_w50": 0.0, "lex.hapax_ratio": 0.0}

        counts = Counter(tokens)
        ttr = len(counts) / len(tokens)
        hapax_ratio = sum(1 for c in counts.values() if c == 1) / len(counts)

        return {
            "lex.ttr": ttr,
            "lex.mattr_w50": self._mattr(tokens, _MATTR_WINDOW),
            "lex.hapax_ratio": hapax_ratio,
        }

    @staticmethod
    def _mattr(tokens: list[str], window: int) -> float:
        if len(tokens) <= window:
            return len(set(tokens)) / len(tokens)
        # Sliding window with incremental counts: O(n) over the document.
        window_counts: Counter[str] = Counter(tokens[:window])
        ttrs = [len(window_counts) / window]
        for i in range(window, len(tokens)):
            outgoing = tokens[i - window]
            window_counts[outgoing] -= 1
            if window_counts[outgoing] == 0:
                del window_counts[outgoing]
            window_counts[tokens[i]] += 1
            ttrs.append(len(window_counts) / window)
        return sum(ttrs) / len(ttrs)
