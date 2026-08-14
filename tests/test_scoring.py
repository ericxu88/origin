"""Scorer interface and stub scorer tests (SPEC AD-5, F-1 interface)."""

from __future__ import annotations

import math

import pytest

from origin_ml.scoring import LN2, FixedScorer, Scorer, StubScorer, perplexity


class TestStubScorer:
    def test_is_a_scorer(self) -> None:
        assert isinstance(StubScorer(), Scorer)
        assert isinstance(FixedScorer({}), Scorer)

    def test_deterministic_across_instances(self) -> None:
        a = StubScorer(seed=7).score("The quick brown fox jumps.")
        b = StubScorer(seed=7).score("The quick brown fox jumps.")
        assert a.tokens == b.tokens

    def test_seed_changes_scores(self) -> None:
        a = StubScorer(seed=1).score("The quick brown fox.")
        b = StubScorer(seed=2).score("The quick brown fox.")
        assert [t.logprob for t in a.tokens] != [t.logprob for t in b.tokens]

    def test_logprob_range_respects_bias_and_spread(self) -> None:
        scored = StubScorer(bias=2.0, spread=3.0).score("alpha beta gamma delta epsilon")
        for token in scored.tokens:
            assert -5.0 <= token.logprob <= -2.0

    def test_offsets_round_trip(self) -> None:
        text = "Origin scores tokens, with offsets."
        for token in StubScorer().score(text).tokens:
            assert text[token.start : token.end] == token.text

    def test_supports_entropy(self) -> None:
        scorer = StubScorer()
        assert scorer.supports_entropy
        scored = scorer.score("hello world")
        assert all(t.entropy is not None for t in scored.tokens)


class TestFixedScorer:
    def test_uses_table_and_default(self) -> None:
        scorer = FixedScorer({"aa": -1.0, "bb": -3.0}, default=-2.0)
        scored = scorer.score("aa bb cc")
        assert [t.logprob for t in scored.tokens] == [-1.0, -3.0, -2.0]

    def test_case_insensitive(self) -> None:
        scorer = FixedScorer({"Word": -1.5})
        assert scorer.score("word WORD Word").logprobs == [-1.5, -1.5, -1.5]

    def test_entropy_optional(self) -> None:
        without = FixedScorer({"a": -1.0})
        assert not without.supports_entropy
        with_entropy = FixedScorer({"a": -1.0}, entropies={"a": 2.5})
        assert with_entropy.supports_entropy
        assert with_entropy.score("a").tokens[0].entropy == 2.5


class TestUnits:
    def test_perplexity_definition(self) -> None:
        # mean logprob = -2.0 → ppl = e^2
        assert perplexity([-1.0, -2.0, -3.0]) == pytest.approx(math.exp(2.0))

    def test_perplexity_empty_is_one(self) -> None:
        assert perplexity([]) == 1.0

    def test_surprisal_bits_conversion(self) -> None:
        token = FixedScorer({"a": -LN2}).score("a").tokens[0]
        # logprob = -ln2 → probability 1/2 → surprisal exactly 1 bit.
        assert token.surprisal_bits == pytest.approx(1.0)

    def test_tokens_in_span_midpoint_rule(self) -> None:
        text = "aa bb cc"
        scored = FixedScorer({}).score(text)
        # Span covering only "bb" (chars 3..5).
        selected = scored.tokens_in_span(3, 5)
        assert [t.text for t in selected] == ["bb"]
