"""Hand-checkable perplexity feature tests (SPEC F-2..F-7, F-13)."""

from __future__ import annotations

import math

import pytest

from origin_ml.features import AnalyzedText, PerplexityExtractor
from origin_ml.scoring import LN2, FixedScorer

# Two sentences, four tokens with exactly known log probabilities.
TEXT = "aa bb. cc dd."
LOGPROBS = {"aa": -1.0, "bb": -2.0, "cc": -3.0, "dd": -4.0}


def extract(entropies: dict[str, float] | None = None) -> dict[str, float]:
    scorer = FixedScorer(LOGPROBS, entropies=entropies)
    doc = AnalyzedText.analyze(TEXT, scorer)
    return PerplexityExtractor().extract(doc)


class TestDocumentLevel:
    def test_doc_perplexity(self) -> None:
        # mean logprob = -2.5 → ppl = e^2.5
        assert extract()["ppl.doc_perplexity"] == pytest.approx(math.exp(2.5))

    def test_mean_token_logprob(self) -> None:
        assert extract()["ppl.mean_token_logprob"] == pytest.approx(-2.5)

    def test_surprisal_stats_in_bits(self) -> None:
        features = extract()
        surprisals = [1.0 / LN2, 2.0 / LN2, 3.0 / LN2, 4.0 / LN2]
        mean = sum(surprisals) / 4
        std = math.sqrt(sum((s - mean) ** 2 for s in surprisals) / 4)
        assert features["ppl.mean_surprisal_bits"] == pytest.approx(mean)
        assert features["ppl.std_surprisal_bits"] == pytest.approx(std)
        assert features["ppl.max_surprisal_bits"] == pytest.approx(4.0 / LN2)


class TestSentenceLevel:
    # Sentence 1 ("aa bb."): mean logprob -1.5 → ppl e^1.5
    # Sentence 2 ("cc dd."): mean logprob -3.5 → ppl e^3.5
    P1 = math.exp(1.5)
    P2 = math.exp(3.5)

    def test_sentence_perplexity_stats(self) -> None:
        features = extract()
        mean = (self.P1 + self.P2) / 2
        std = abs(self.P2 - self.P1) / 2
        assert features["ppl.sent_ppl_mean"] == pytest.approx(mean)
        assert features["ppl.sent_ppl_std"] == pytest.approx(std)
        assert features["ppl.sent_ppl_var"] == pytest.approx(std**2)
        assert features["ppl.sent_ppl_range"] == pytest.approx(self.P2 - self.P1)
        assert features["ppl.sent_ppl_min"] == pytest.approx(self.P1)
        assert features["ppl.sent_ppl_max"] == pytest.approx(self.P2)

    def test_sentence_perplexity_cv_and_burstiness(self) -> None:
        features = extract()
        mean = (self.P1 + self.P2) / 2
        std = abs(self.P2 - self.P1) / 2
        assert features["ppl.sent_ppl_cv"] == pytest.approx(std / mean)
        assert features["ppl.sent_ppl_burstiness"] == pytest.approx((std - mean) / (std + mean))


class TestEntropy:
    def test_entropy_stats_when_supported(self) -> None:
        features = extract(entropies={"aa": 1.0, "bb": 2.0, "cc": 3.0, "dd": 4.0})
        assert features["ppl.has_entropy"] == 1.0
        assert features["ppl.entropy_mean"] == pytest.approx(2.5)
        assert features["ppl.entropy_std"] == pytest.approx(math.sqrt(1.25))

    def test_graceful_degradation_without_entropy(self) -> None:
        features = extract(entropies=None)
        assert features["ppl.has_entropy"] == 0.0
        assert features["ppl.entropy_mean"] == 0.0
        assert features["ppl.entropy_std"] == 0.0


class TestDegenerateInput:
    def test_empty_text_produces_finite_defaults(self) -> None:
        doc = AnalyzedText.analyze("", FixedScorer({}))
        features = PerplexityExtractor().extract(doc)
        assert features["ppl.doc_perplexity"] == 1.0
        assert all(math.isfinite(v) for v in features.values())

    def test_requires_scoring(self) -> None:
        doc = AnalyzedText.analyze("hello", scorer=None)
        with pytest.raises(ValueError, match="requires a scored document"):
            PerplexityExtractor().extract(doc)
