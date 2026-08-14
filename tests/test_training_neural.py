"""Neural training loop tests on the tiny fixture (SPEC N-4, N-6)."""

from __future__ import annotations

from pathlib import Path

import pytest

from origin_ml.datasets import DocLabel, read_jsonl
from origin_ml.detectors import NeuralDetector
from origin_ml.training import NeuralTrainConfig, sentence_examples, train_neural

TINY_CLASSIFIER = Path(__file__).resolve().parent / "fixtures" / "tiny_classifier"
SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample" / "documents.jsonl"


class TestSentenceExamples:
    def test_pure_documents_labelled_uniformly(self) -> None:
        records = [r for r in read_jsonl(SAMPLE) if r.label is not DocLabel.MIXED][:6]
        examples = sentence_examples(records)
        assert examples
        by_label = {0: 0, 1: 0}
        for _, label in examples:
            by_label[label] += 1
        human_sentences = sum(
            1 for r in records if r.label is DocLabel.HUMAN for _ in r.text.split(".") if _
        )
        assert by_label[0] > 0 or human_sentences == 0

    def test_mixed_documents_use_span_labels(self) -> None:
        mixed = [r for r in read_jsonl(SAMPLE) if r.label is DocLabel.MIXED][:3]
        assert mixed
        examples = sentence_examples(mixed)
        labels = {label for _, label in examples}
        assert labels == {0, 1}  # both classes present inside mixed docs


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> tuple[NeuralDetector, object]:
    records = read_jsonl(SAMPLE)
    subset = [r for r in records if r.label is DocLabel.HUMAN][:8] + [
        r for r in records if r.label is DocLabel.AI
    ][:8]
    config = NeuralTrainConfig(
        checkpoint=str(TINY_CLASSIFIER),
        output_dir=tmp_path_factory.mktemp("ckpt") / "trained",
        epochs=3,
        lr=5e-4,
        batch_size=8,
        seed=0,
        device="cpu",
    )
    return train_neural(subset, config)


class TestTrainNeural:
    def test_loss_decreases(self, trained: tuple[NeuralDetector, object]) -> None:
        _, report = trained
        losses = report.epoch_losses  # type: ignore[attr-defined]
        assert len(losses) == 3
        assert losses[-1] < losses[0]

    def test_report_metadata(self, trained: tuple[NeuralDetector, object]) -> None:
        _, report = trained
        assert report.n_examples > 50  # type: ignore[attr-defined]
        assert report.device == "cpu"  # type: ignore[attr-defined]
        assert report.saved_to is not None  # type: ignore[attr-defined]

    def test_checkpoint_reloads_with_same_predictions(
        self, trained: tuple[NeuralDetector, object]
    ) -> None:
        detector, report = trained
        loaded = NeuralDetector.load(Path(report.saved_to), device="cpu")  # type: ignore[attr-defined]
        sentences = ["Additionally, the system provides a clear approach to results."]
        assert detector.sentence_probs(sentences) == pytest.approx(
            loaded.sentence_probs(sentences), abs=1e-6
        )

    def test_training_actually_learned_the_fixture_classes(
        self, trained: tuple[NeuralDetector, object]
    ) -> None:
        """After training, formulaic 'AI' fixture text scores higher than bursty text."""
        detector, _ = trained
        ai_like = "Additionally, the system provides a clear and effective way to improve results."
        human_like = "Rain again. Who remembers the crooked lantern now?"
        p_ai, p_human = detector.sentence_probs([ai_like, human_like])
        assert p_ai > p_human

    def test_rejects_empty_records(self) -> None:
        with pytest.raises(ValueError, match="no training sentences"):
            train_neural([], NeuralTrainConfig(checkpoint=str(TINY_CLASSIFIER), device="cpu"))
