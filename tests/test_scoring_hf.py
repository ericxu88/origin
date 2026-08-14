"""HFCausalScorer tests against the committed tiny causal LM (SPEC F-1, N-6)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from origin_ml.scoring import Scorer, perplexity
from origin_ml.scoring.hf import HFCausalScorer

TINY_LM = Path(__file__).resolve().parent / "fixtures" / "tiny_causal_lm"


@pytest.fixture(scope="module")
def scorer() -> HFCausalScorer:
    return HFCausalScorer(str(TINY_LM), device="cpu")


TEXT = "The people said the world would change and the work would continue."


class TestHFCausalScorer:
    def test_is_a_scorer(self, scorer: HFCausalScorer) -> None:
        assert isinstance(scorer, Scorer)
        assert scorer.name == f"hf-causal({TINY_LM})"

    def test_offsets_round_trip(self, scorer: HFCausalScorer) -> None:
        scored = scorer.score(TEXT)
        assert scored.tokens
        for token in scored.tokens:
            assert TEXT[token.start : token.end] == token.text

    def test_first_token_not_scored(self, scorer: HFCausalScorer) -> None:
        scored = scorer.score(TEXT)
        assert all(token.start > 0 for token in scored.tokens)

    def test_logprobs_are_negative_and_finite(self, scorer: HFCausalScorer) -> None:
        scored = scorer.score(TEXT)
        for token in scored.tokens:
            assert token.logprob < 0.0
            assert math.isfinite(token.logprob)

    def test_entropy_supported_and_positive(self, scorer: HFCausalScorer) -> None:
        assert scorer.supports_entropy
        scored = scorer.score(TEXT)
        for token in scored.tokens:
            assert token.entropy is not None
            assert token.entropy > 0.0

    def test_entropy_can_be_disabled(self) -> None:
        scorer = HFCausalScorer(str(TINY_LM), device="cpu", compute_entropy=False)
        assert not scorer.supports_entropy
        assert all(t.entropy is None for t in scorer.score(TEXT).tokens)

    def test_deterministic(self, scorer: HFCausalScorer) -> None:
        assert scorer.score(TEXT) == scorer.score(TEXT)

    def test_perplexity_finite(self, scorer: HFCausalScorer) -> None:
        ppl = perplexity(scorer.score(TEXT).logprobs)
        assert math.isfinite(ppl)
        assert ppl > 1.0

    def test_empty_and_single_token_inputs(self, scorer: HFCausalScorer) -> None:
        assert scorer.score("").tokens == ()
        assert scorer.score("   ").tokens == ()
        assert scorer.score("word").tokens == ()  # no left context to condition on

    def test_truncation_respects_max_length(self) -> None:
        scorer = HFCausalScorer(str(TINY_LM), device="cpu", max_length=8)
        long_text = " ".join(["word"] * 50)
        assert len(scorer.score(long_text).tokens) <= 7

    def test_works_in_feature_pipeline(self, scorer: HFCausalScorer) -> None:
        from origin_ml.features import build_default_pipeline

        pipeline = build_default_pipeline(scorer=scorer)
        vector = pipeline.extract("The world would change. And the work would continue.")
        assert vector["ppl.doc_perplexity"] > 1.0
        assert vector["ppl.has_entropy"] == 1.0
