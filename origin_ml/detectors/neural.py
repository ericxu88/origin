"""Neural detector: sentence-granular transformer classification (SPEC §3.3).

Architecture (SPEC AD-7): a Hugging Face sequence-classification encoder with a
configurable checkpoint classifies each *sentence*; document-level Human/AI/
Mixed labels come from the shared aggregation rule
(:mod:`origin_ml.detectors.aggregation`). This gives localization for free and
makes the document decision auditable.

Label convention: binary head with index 1 = "ai". When a checkpoint's
``id2label`` names a label ``ai`` (any case), that index is used instead, so
externally trained checkpoints with swapped label order still work.

Runs on CPU; uses CUDA/MPS automatically when available (SPEC N-5).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from origin_ml.datasets.schema import DocLabel
from origin_ml.detectors.aggregation import (
    AggregationConfig,
    DocumentDecision,
    aggregate_sentence_probs,
)
from origin_ml.device import resolve_device
from origin_ml.text.segmentation import segment_sentences

__all__ = ["NeuralDetector", "NeuralPrediction", "SentenceProbability"]


@dataclass(frozen=True)
class SentenceProbability:
    """Per-sentence AI probability with exact document offsets (SPEC N-3, L-2)."""

    text: str
    start: int
    end: int
    p_ai: float


@dataclass(frozen=True)
class NeuralPrediction:
    """Structured document prediction (SPEC N-2)."""

    label: DocLabel
    confidence: float
    mean_p_ai: float
    frac_ai_sentences: float
    sentences: tuple[SentenceProbability, ...]


def _ai_label_index(id2label: dict[int, str] | None) -> int:
    if id2label:
        for index, name in id2label.items():
            if str(name).strip().lower() == "ai":
                return int(index)
    return 1


class NeuralDetector:
    """Sentence-level transformer classifier with document aggregation."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: object,
        *,
        checkpoint: str,
        device: str | None = None,
        max_length: int = 256,
        batch_size: int = 16,
        aggregation: AggregationConfig | None = None,
    ) -> None:
        self._device = resolve_device(device)
        self._model = model.to(self._device).eval()
        self._tokenizer = tokenizer
        self.checkpoint = checkpoint
        self._max_length = max_length
        self._batch_size = batch_size
        self.aggregation = aggregation or AggregationConfig()
        config = getattr(model, "config", None)
        id2label = getattr(config, "id2label", None)
        self._ai_index = _ai_label_index(
            {int(k): str(v) for k, v in id2label.items()} if id2label else None
        )
        num_labels = int(getattr(config, "num_labels", 2))
        if num_labels != 2:
            raise ValueError(f"neural detector requires a binary head, got {num_labels} labels")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        device: str | None = None,
        max_length: int = 256,
        batch_size: int = 16,
        aggregation: AggregationConfig | None = None,
    ) -> NeuralDetector:
        """Load any local path or hub id (SPEC N-1); works fully offline for paths."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        checkpoint = str(checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
        return cls(
            model,
            tokenizer,
            checkpoint=checkpoint,
            device=device,
            max_length=max_length,
            batch_size=batch_size,
            aggregation=aggregation,
        )

    @property
    def device(self) -> torch.device:
        return self._device

    def sentence_probs(self, sentences: list[str]) -> list[float]:
        """``P(ai)`` per sentence, batched (SPEC N-3)."""
        if not sentences:
            return []
        probs: list[float] = []
        tokenize = self._tokenizer  # PreTrainedTokenizerBase is callable
        for i in range(0, len(sentences), self._batch_size):
            batch = sentences[i : i + self._batch_size]
            encoding = tokenize(  # type: ignore[operator]
                batch,
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            )
            encoding = {k: v.to(self._device) for k, v in encoding.items()}
            with torch.no_grad():
                logits = self._model(**encoding).logits
            batch_probs = torch.softmax(logits.float(), dim=-1)[:, self._ai_index]
            probs.extend(float(p) for p in batch_probs.to("cpu"))
        return probs

    def predict_document(self, text: str) -> NeuralPrediction:
        """Segment, classify sentences, aggregate to Human/AI/Mixed (SPEC N-2)."""
        spans = segment_sentences(text)
        if not spans:
            raise ValueError("document contains no sentences to classify")
        probs = self.sentence_probs([span.text for span in spans])
        decision: DocumentDecision = aggregate_sentence_probs(probs, self.aggregation)
        return NeuralPrediction(
            label=decision.label,
            confidence=decision.confidence,
            mean_p_ai=decision.mean_p_ai,
            frac_ai_sentences=decision.frac_ai_sentences,
            sentences=tuple(
                SentenceProbability(text=span.text, start=span.start, end=span.end, p_ai=p)
                for span, p in zip(spans, probs, strict=True)
            ),
        )

    def save(self, directory: Path) -> None:
        """Persist model + tokenizer as a standard HF checkpoint (SPEC N-4)."""
        directory.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(directory)  # type: ignore[operator]
        self._tokenizer.save_pretrained(directory)  # type: ignore[attr-defined]

    @classmethod
    def load(cls, directory: Path, *, device: str | None = None) -> NeuralDetector:
        return cls.from_checkpoint(directory, device=device)
