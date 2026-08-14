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
    """Decision rule v2: doc_p-driven verdict + band/dispersion mixture test."""

    def test_ai_via_mean_fallback(self) -> None:
        # No doc_p: edp = mean = 0.9 -> above the mixed band -> AI.
        decision = aggregate_sentence_probs([0.9, 0.95, 0.85])
        assert decision.label is DocLabel.AI
        assert decision.p_mixed == 0.0
        assert decision.p_ai == pytest.approx(0.9)
        assert decision.confidence == pytest.approx(0.9)

    def test_human_via_mean_fallback(self) -> None:
        decision = aggregate_sentence_probs([0.1, 0.05, 0.2])
        assert decision.label is DocLabel.HUMAN
        assert decision.p_human == pytest.approx(1 - (0.1 + 0.05 + 0.2) / 3)

    def test_bimodal_sentences_are_mixed(self) -> None:
        # edp = mean = 0.5 (in band [0.45, 0.85]), dispersion = 0.4 >= 0.1 -> MIXED.
        # h = (0.85 - 0.45) / 4 = 0.1; depth = min(0.05, 0.35) / 0.1 = 0.5.
        decision = aggregate_sentence_probs([0.9, 0.9, 0.1, 0.1])
        assert decision.label is DocLabel.MIXED
        assert decision.frac_ai_sentences == 0.5
        assert decision.p_mixed == pytest.approx(0.7)
        assert decision.p_ai == pytest.approx(0.15)
        assert decision.p_human == pytest.approx(0.15)

    def test_uniform_ambiguity_is_not_mixed(self) -> None:
        # Mid probability with no dispersion is ambiguity, not mixture.
        decision = aggregate_sentence_probs([0.5, 0.5, 0.5])
        assert decision.label is not DocLabel.MIXED
        assert decision.p_mixed == 0.0

    def test_doc_p_overrides_sentence_bias(self) -> None:
        """The human-corpus fix: mildly AI-ish sentences, confident human doc_p."""
        decision = aggregate_sentence_probs([0.6, 0.6, 0.6], doc_p=0.05)
        assert decision.label is DocLabel.HUMAN
        assert decision.p_human == pytest.approx(0.95)

    def test_doc_p_in_band_with_dispersion_is_mixed(self) -> None:
        decision = aggregate_sentence_probs([0.9, 0.1], doc_p=0.6)
        assert decision.label is DocLabel.MIXED
        assert decision.p_mixed == pytest.approx(0.9)  # depth caps at 1

    def test_partial_band_depth(self) -> None:
        # h = (0.85 - 0.45) / 4 = 0.1; edp = 0.5 -> depth = 0.05 / 0.1 = 0.5.
        decision = aggregate_sentence_probs([0.9, 0.1], doc_p=0.5)
        assert decision.label is DocLabel.MIXED
        assert decision.p_mixed == pytest.approx(0.7)

    def test_below_band_is_not_mixed(self) -> None:
        # doc_p below the band -> human even with dispersed sentences.
        decision = aggregate_sentence_probs([0.9, 0.1], doc_p=0.4)
        assert decision.label is DocLabel.HUMAN

    def test_class_scores_sum_to_one_and_match_label(self) -> None:
        cases = [
            ([0.9, 0.95, 0.85], None),
            ([0.1, 0.05, 0.2], None),
            ([0.9, 0.9, 0.1, 0.1], None),
            ([0.6, 0.6, 0.6], 0.05),
            ([0.9, 0.1], 0.6),
            ([0.2, 0.9], 0.95),
        ]
        for probs, doc_p in cases:
            decision = aggregate_sentence_probs(probs, doc_p=doc_p)
            scores = {"human": decision.p_human, "ai": decision.p_ai, "mixed": decision.p_mixed}
            assert sum(scores.values()) == pytest.approx(1.0)
            assert max(scores, key=lambda k: scores[k]) == decision.label.value
            assert decision.confidence == pytest.approx(max(scores.values()))

    def test_custom_band(self) -> None:
        config = AggregationConfig(mixed_band_low=0.05, mixed_band_high=0.95)
        decision = aggregate_sentence_probs([0.9, 0.1], config, doc_p=0.9)
        assert decision.label is DocLabel.MIXED

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            aggregate_sentence_probs([])

    def test_bad_config_rejected(self) -> None:
        with pytest.raises(ValueError, match="mixed_band_low"):
            AggregationConfig(mixed_band_low=0.9, mixed_band_high=0.1)
        with pytest.raises(ValueError, match="dispersion"):
            AggregationConfig(mixed_min_dispersion=-1.0)
