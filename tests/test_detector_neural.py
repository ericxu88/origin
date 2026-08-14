"""Neural detector tests against the committed tiny classifier (SPEC N-1..N-6)."""

from __future__ import annotations

from pathlib import Path

import pytest

from origin_ml.datasets import DocLabel
from origin_ml.detectors import AggregationConfig, NeuralDetector, aggregate_sentence_probs

TINY_CLASSIFIER = Path(__file__).resolve().parent / "fixtures" / "tiny_classifier"


@pytest.fixture(scope="module")
def detector() -> NeuralDetector:
    return NeuralDetector.from_checkpoint(TINY_CLASSIFIER, device="cpu")


TEXT = "The people said the world would change. The work would continue as before."


class TestSentenceProbs:
    def test_probabilities_in_range(self, detector: NeuralDetector) -> None:
        probs = detector.sentence_probs(["First sentence here.", "Second sentence here."])
        assert len(probs) == 2
        assert all(0.0 <= p <= 1.0 for p in probs)

    def test_empty_input(self, detector: NeuralDetector) -> None:
        assert detector.sentence_probs([]) == []

    def test_batching_matches_single(self, detector: NeuralDetector) -> None:
        sentences = [f"Sentence number {i} talks about the world." for i in range(5)]
        small_batch = NeuralDetector.from_checkpoint(TINY_CLASSIFIER, device="cpu", batch_size=2)
        a = detector.sentence_probs(sentences)
        b = small_batch.sentence_probs(sentences)
        assert a == pytest.approx(b, abs=1e-5)

    def test_deterministic(self, detector: NeuralDetector) -> None:
        sentences = ["The same sentence yields the same probability."]
        assert detector.sentence_probs(sentences) == detector.sentence_probs(sentences)


class TestPredictDocument:
    def test_structured_output_with_offsets(self, detector: NeuralDetector) -> None:
        prediction = detector.predict_document(TEXT)
        assert prediction.label in (DocLabel.HUMAN, DocLabel.AI, DocLabel.MIXED)
        assert 0.0 <= prediction.confidence <= 1.0
        assert len(prediction.sentences) == 2
        for sentence in prediction.sentences:
            assert TEXT[sentence.start : sentence.end] == sentence.text
            assert 0.0 <= sentence.p_ai <= 1.0

    def test_empty_document_rejected(self, detector: NeuralDetector) -> None:
        with pytest.raises(ValueError, match="no sentences"):
            detector.predict_document("   ")


class TestSaveLoad:
    def test_round_trip_predictions_identical(
        self, detector: NeuralDetector, tmp_path: Path
    ) -> None:
        target = tmp_path / "checkpoint"
        detector.save(target)
        loaded = NeuralDetector.load(target, device="cpu")
        sentences = ["A sentence to compare probabilities on."]
        assert detector.sentence_probs(sentences) == pytest.approx(
            loaded.sentence_probs(sentences), abs=1e-6
        )


class TestAggregation:
    def test_all_ai_sentences(self) -> None:
        decision = aggregate_sentence_probs([0.9, 0.95, 0.85])
        assert decision.label is DocLabel.AI
        assert decision.confidence == pytest.approx((0.9 + 0.95 + 0.85) / 3)

    def test_all_human_sentences(self) -> None:
        decision = aggregate_sentence_probs([0.1, 0.05, 0.2])
        assert decision.label is DocLabel.HUMAN
        assert decision.confidence == pytest.approx(1 - (0.1 + 0.05 + 0.2) / 3)

    def test_half_and_half_is_mixed(self) -> None:
        decision = aggregate_sentence_probs([0.9, 0.9, 0.1, 0.1])
        assert decision.label is DocLabel.MIXED
        assert decision.frac_ai_sentences == 0.5
        assert decision.confidence == pytest.approx(0.8)  # mean 2|p - 0.5|

    def test_custom_thresholds(self) -> None:
        config = AggregationConfig(mixed_low=0.05, mixed_high=0.95)
        decision = aggregate_sentence_probs([0.9, 0.9, 0.9, 0.1], config)
        assert decision.label is DocLabel.MIXED

    def test_class_scores_sum_to_one_and_match_label(self) -> None:
        for probs in ([0.9, 0.95, 0.85], [0.1, 0.05, 0.2], [0.9, 0.9, 0.1, 0.1], [0.6, 0.4]):
            decision = aggregate_sentence_probs(probs)
            scores = {"human": decision.p_human, "ai": decision.p_ai, "mixed": decision.p_mixed}
            assert sum(scores.values()) == pytest.approx(1.0)
            assert max(scores, key=lambda k: scores[k]) == decision.label.value

    def test_class_scores_hand_computed(self) -> None:
        # Defaults: low=0.25, high=0.75 -> transition half-width h=0.125.
        # frac_ai = 1.0 -> pure AI zone.
        decision = aggregate_sentence_probs([0.9, 0.9])
        assert (decision.p_human, decision.p_mixed, decision.p_ai) == (0.0, 0.0, 1.0)
        # frac_ai = 0.0 -> pure human zone.
        decision = aggregate_sentence_probs([0.1, 0.2])
        assert (decision.p_human, decision.p_mixed, decision.p_ai) == (1.0, 0.0, 0.0)
        # frac_ai = 0.5 -> mixed plateau.
        decision = aggregate_sentence_probs([0.9, 0.1])
        assert decision.p_mixed == 1.0
        # frac_ai = 0.3 -> inside the lower transition [0.125, 0.375]:
        # mixed = (0.3 - 0.125) / 0.25 = 0.7, human = 0.3.
        decision = aggregate_sentence_probs([0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        assert decision.p_mixed == pytest.approx(0.7)
        assert decision.p_human == pytest.approx(0.3)
        assert decision.p_ai == 0.0
        assert decision.label is DocLabel.MIXED  # coherent with hard rule

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            aggregate_sentence_probs([])

    def test_bad_config_rejected(self) -> None:
        with pytest.raises(ValueError, match="mixed_low"):
            AggregationConfig(mixed_low=0.9, mixed_high=0.1)
