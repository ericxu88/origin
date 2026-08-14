"""Sentence segmentation tests (SPEC L-1, AD-6)."""

from __future__ import annotations

from itertools import pairwise

import pytest

from origin_ml.text import segment_sentences


def texts_of(text: str) -> list[str]:
    return [s.text for s in segment_sentences(text)]


class TestRoundTrip:
    """Offsets must reproduce the sentence text exactly (L-1)."""

    @pytest.mark.parametrize(
        "text",
        [
            "One sentence.",
            "First. Second! Third?",
            "  Leading space. Trailing space.  ",
            "Dr. Smith visited Mrs. Jones. They talked.",
            "Value is 3.14 today. Tomorrow it is 2.71.",
            "Para one line one\ncontinues here.\n\nPara two starts. And ends!",
            'He said "Stop." Then he left.',
            "No terminator at all",
            "Multiple?! Terminators... And more.",
            "",
            "   \n\n  \t ",
        ],
    )
    def test_offsets_round_trip(self, text: str) -> None:
        for span in segment_sentences(text):
            assert text[span.start : span.end] == span.text

    def test_spans_are_ordered_and_disjoint(self) -> None:
        text = "First. Second! Third? Fourth."
        spans = segment_sentences(text)
        for a, b in pairwise(spans):
            assert a.end <= b.start


class TestBoundaries:
    def test_simple_split(self) -> None:
        assert texts_of("First. Second! Third?") == ["First.", "Second!", "Third?"]

    def test_abbreviations_do_not_split(self) -> None:
        assert texts_of("Dr. Smith arrived. He sat down.") == [
            "Dr. Smith arrived.",
            "He sat down.",
        ]

    def test_initials_do_not_split(self) -> None:
        assert texts_of("J. K. Rowling wrote it. It sold well.") == [
            "J. K. Rowling wrote it.",
            "It sold well.",
        ]

    def test_dotted_initialisms_do_not_split(self) -> None:
        assert texts_of("Use e.g. apples. Or i.e. fruit.") == [
            "Use e.g. apples.",
            "Or i.e. fruit.",
        ]

    def test_decimals_do_not_split(self) -> None:
        assert texts_of("Pi is 3.14 exactly. No it is not.") == [
            "Pi is 3.14 exactly.",
            "No it is not.",
        ]

    def test_urls_mid_token_do_not_split(self) -> None:
        assert texts_of("See example.com for info. Thanks.") == [
            "See example.com for info.",
            "Thanks.",
        ]

    def test_paragraph_break_is_boundary_without_terminator(self) -> None:
        assert texts_of("A heading without period\n\nBody sentence.") == [
            "A heading without period",
            "Body sentence.",
        ]

    def test_single_newline_is_not_boundary(self) -> None:
        assert texts_of("A sentence wrapping\nacross lines.") == [
            "A sentence wrapping\nacross lines."
        ]

    def test_closing_quote_belongs_to_sentence(self) -> None:
        assert texts_of('He said "Stop." Then he left.') == ['He said "Stop."', "Then he left."]

    def test_multi_terminators(self) -> None:
        assert texts_of("What?! Really... Yes.") == ["What?!", "Really...", "Yes."]

    def test_empty_and_blank(self) -> None:
        assert texts_of("") == []
        assert texts_of("  \n\n \t ") == []

    def test_no_terminator_yields_single_sentence(self) -> None:
        assert texts_of("no terminator here") == ["no terminator here"]
