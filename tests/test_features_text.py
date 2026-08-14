"""Hand-checkable tests for scorer-free features (SPEC F-6, F-8..F-11, F-13)."""

from __future__ import annotations

import math

import pytest

from origin_ml.features import (
    AnalyzedText,
    LexicalDiversityExtractor,
    RepetitionExtractor,
    SentenceStatsExtractor,
    StyleExtractor,
)


def analyze(text: str) -> AnalyzedText:
    return AnalyzedText.analyze(text)


class TestSentenceStats:
    def test_two_sentence_document(self) -> None:
        # "One two three." → 3 words, 14 chars; "Four five." → 2 words, 10 chars.
        features = SentenceStatsExtractor().extract(analyze("One two three. Four five."))
        assert features["sent.count"] == 2.0
        assert features["sent.mean_len_words"] == pytest.approx(2.5)
        assert features["sent.std_len_words"] == pytest.approx(0.5)
        assert features["sent.min_len_words"] == 2.0
        assert features["sent.max_len_words"] == 3.0
        assert features["sent.mean_len_chars"] == pytest.approx(12.0)
        assert features["sent.std_len_chars"] == pytest.approx(2.0)
        assert features["sent.len_words_cv"] == pytest.approx(0.5 / 2.5)
        assert features["sent.len_words_burstiness"] == pytest.approx((0.5 - 2.5) / (0.5 + 2.5))

    def test_empty_document(self) -> None:
        features = SentenceStatsExtractor().extract(analyze(""))
        assert features["sent.count"] == 0.0
        assert all(math.isfinite(v) for v in features.values())


class TestLexicalDiversity:
    def test_ttr_and_hapax(self) -> None:
        # tokens: aa bb aa cc → 4 tokens, 3 types; hapax types: bb, cc.
        features = LexicalDiversityExtractor().extract(analyze("aa bb aa cc"))
        assert features["lex.ttr"] == pytest.approx(3 / 4)
        assert features["lex.hapax_ratio"] == pytest.approx(2 / 3)

    def test_mattr_equals_ttr_below_window(self) -> None:
        features = LexicalDiversityExtractor().extract(analyze("aa bb aa cc"))
        assert features["lex.mattr_w50"] == pytest.approx(features["lex.ttr"])

    def test_mattr_windowed(self) -> None:
        # 60 tokens alternating "a b": every 50-token window has 2 types.
        text = " ".join(["a", "b"] * 30)
        features = LexicalDiversityExtractor().extract(analyze(text))
        assert features["lex.mattr_w50"] == pytest.approx(2 / 50)

    def test_case_insensitive(self) -> None:
        features = LexicalDiversityExtractor().extract(analyze("Word word WORD"))
        assert features["lex.ttr"] == pytest.approx(1 / 3)

    def test_empty(self) -> None:
        features = LexicalDiversityExtractor().extract(analyze(""))
        assert features == {"lex.ttr": 0.0, "lex.mattr_w50": 0.0, "lex.hapax_ratio": 0.0}


class TestRepetition:
    def test_distinct_ngrams(self) -> None:
        # tokens: aa bb aa bb
        features = RepetitionExtractor().extract(analyze("aa bb aa bb"))
        assert features["rep.distinct_1"] == pytest.approx(2 / 4)
        # bigrams: (aa,bb) (bb,aa) (aa,bb) → 2 distinct of 3
        assert features["rep.distinct_2"] == pytest.approx(2 / 3)
        assert features["rep.top_bigram_share"] == pytest.approx(2 / 3)
        # trigrams: (aa,bb,aa) (bb,aa,bb) → all distinct
        assert features["rep.distinct_3"] == pytest.approx(1.0)
        assert features["rep.top_trigram_share"] == pytest.approx(1 / 2)

    def test_repeated_sentence_starts(self) -> None:
        text = "Also one. Also two. Then three."
        features = RepetitionExtractor().extract(analyze(text))
        # first words: also, also, then → 1 - 2/3
        assert features["rep.repeated_sentence_start_ratio"] == pytest.approx(1 / 3)

    def test_empty(self) -> None:
        features = RepetitionExtractor().extract(analyze(""))
        assert features["rep.distinct_1"] == 1.0
        assert features["rep.top_bigram_share"] == 0.0
        assert features["rep.repeated_sentence_start_ratio"] == 0.0


class TestStyle:
    TEXT = "The cat, the DOG; 42."
    # words: The, cat, the, DOG, 42 → 5 tokens
    # non-space chars: "Thecat,theDOG;42." → 17
    # punctuation chars: , ; . → 3; digit chars: 4 2 → 2

    def test_style_features(self) -> None:
        features = StyleExtractor().extract(analyze(self.TEXT))
        assert features["style.stopword_ratio"] == pytest.approx(2 / 5)
        assert features["style.common_word_ratio"] == pytest.approx(2 / 5)
        assert features["style.avg_word_length"] == pytest.approx((3 + 3 + 3 + 3 + 2) / 5)
        assert features["style.punct_char_ratio"] == pytest.approx(3 / 17)
        assert features["style.comma_rate"] == pytest.approx(1 / 5)
        assert features["style.semicolon_rate"] == pytest.approx(1 / 5)
        assert features["style.digit_char_ratio"] == pytest.approx(2 / 17)
        assert features["style.allcaps_word_ratio"] == pytest.approx(1 / 5)

    def test_empty(self) -> None:
        features = StyleExtractor().extract(analyze(""))
        assert all(v == 0.0 for v in features.values())
