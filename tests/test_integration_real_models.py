"""Opt-in integration tests with real checkpoints (SPEC N-6, Q-2).

Skipped by default (pytest addopts excludes the ``integration`` marker) and
under offline mode. To run::

    HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 uv run pytest -m integration

These download small real models (distilgpt2 ≈ 350 MB) on first run.
"""

from __future__ import annotations

import math
import os

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("HF_HUB_OFFLINE") == "1",
        reason="requires network access (set HF_HUB_OFFLINE=0 to run)",
    ),
]


class TestRealCausalScorer:
    def test_distilgpt2_scores_sensibly(self) -> None:
        from origin_ml.scoring.hf import HFCausalScorer

        scorer = HFCausalScorer("distilgpt2", device="cpu")
        scored = scorer.score("The quick brown fox jumps over the lazy dog.")
        assert scored.tokens
        for token in scored.tokens:
            assert token.logprob < 0.0
            assert math.isfinite(token.logprob)
        # A common continuation should be far more probable than a bizarre one.
        common = scorer.score("New York City is the largest city in the United States.")
        rare = scorer.score("Purple calculus trombone yesterday vaccinates the moon.")
        from origin_ml.scoring import perplexity

        assert perplexity(common.logprobs) < perplexity(rare.logprobs)


class TestRealNeuralDetector:
    def test_distilbert_loads_and_predicts(self) -> None:
        from origin_ml.detectors import NeuralDetector

        detector = NeuralDetector.from_checkpoint("distilbert-base-uncased", device="cpu")
        prediction = detector.predict_document(
            "This is one sentence. This is another sentence entirely."
        )
        assert len(prediction.sentences) == 2
        for sentence in prediction.sentences:
            assert 0.0 <= sentence.p_ai <= 1.0
