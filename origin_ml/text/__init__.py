"""Deterministic text utilities: sentence segmentation and word tokenization."""

from origin_ml.text.segmentation import SentenceSpan, segment_sentences
from origin_ml.text.words import WordSpan, tokenize_words

__all__ = ["SentenceSpan", "WordSpan", "segment_sentences", "tokenize_words"]
