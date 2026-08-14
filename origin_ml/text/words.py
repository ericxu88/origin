"""Word tokenization with character offsets (deterministic, regex-based)."""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["WordSpan", "tokenize_words"]


@dataclass(frozen=True, slots=True)
class WordSpan:
    """A word token with exact character offsets into the source document."""

    text: str
    start: int
    end: int


# Alphabetic words may contain internal apostrophes ("don't", "o'clock");
# numbers may contain internal decimal points or thousands separators.
_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*|\d+(?:[.,]\d+)*")


def tokenize_words(text: str) -> list[WordSpan]:
    """Extract word tokens with offsets; ``text[start:end]`` round-trips."""
    return [WordSpan(text=m.group(), start=m.start(), end=m.end()) for m in _WORD.finditer(text)]
