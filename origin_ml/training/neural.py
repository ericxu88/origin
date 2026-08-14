"""Configurable neural training loop (SPEC N-4).

Trains the sentence-granular neural detector on dataset records:

- human documents contribute their sentences with label 0,
- ai documents contribute their sentences with label 1,
- mixed documents contribute per-sentence labels derived from their ground
  truth spans (sentence midpoint inside an AI span → label 1).

Determinism: Python/NumPy/Torch RNGs are seeded from the config; data order is
driven by a seeded ``torch.Generator``. CPU training of the tiny test fixture
takes seconds; real checkpoints use the same code path (SPEC N-5, N-6).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from origin_ml.datasets.schema import DocLabel, DocumentRecord, SegmentLabel
from origin_ml.detectors.neural import NeuralDetector
from origin_ml.device import resolve_device
from origin_ml.text.segmentation import segment_sentences

__all__ = ["NeuralTrainConfig", "TrainReport", "sentence_examples", "train_neural"]


@dataclass(frozen=True)
class NeuralTrainConfig:
    """Everything the training loop needs (SPEC N-4)."""

    checkpoint: str
    output_dir: Path | None = None
    epochs: int = 2
    lr: float = 5e-4
    batch_size: int = 8
    max_length: int = 128
    seed: int = 0
    device: str | None = None


@dataclass(frozen=True)
class TrainReport:
    """What actually happened during training."""

    epoch_losses: tuple[float, ...] = field(default=())
    n_examples: int = 0
    device: str = "cpu"
    saved_to: str | None = None


def sentence_examples(records: list[DocumentRecord]) -> list[tuple[str, int]]:
    """Flatten records into (sentence, label) pairs; see module docstring."""
    examples: list[tuple[str, int]] = []
    for record in records:
        for span in segment_sentences(record.text):
            if record.label is DocLabel.HUMAN:
                label = 0
            elif record.label is DocLabel.AI:
                label = 1
            else:
                midpoint = (span.start + span.end) / 2
                label = int(
                    any(
                        s.label is SegmentLabel.AI and s.start <= midpoint < s.end
                        for s in record.spans
                    )
                )
            examples.append((span.text, label))
    return examples


class _SentenceDataset(Dataset[tuple[str, int]]):
    def __init__(self, examples: list[tuple[str, int]]) -> None:
        self._examples = examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> tuple[str, int]:
        return self._examples[index]


def train_neural(
    records: list[DocumentRecord], config: NeuralTrainConfig
) -> tuple[NeuralDetector, TrainReport]:
    """Fine-tune a sequence classifier on sentence examples (SPEC N-4)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    examples = sentence_examples(records)
    if not examples:
        raise ValueError("no training sentences derived from the given records")

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = resolve_device(config.device)
    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.checkpoint,
        num_labels=2,
        id2label={0: "human", 1: "ai"},
        label2id={"human": 0, "ai": 1},
    ).to(device)
    model.train()

    def collate(batch: list[tuple[str, int]]) -> dict[str, torch.Tensor]:
        texts = [text for text, _ in batch]
        labels = torch.tensor([label for _, label in batch], dtype=torch.long)
        encoding = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=config.max_length,
            return_tensors="pt",
        )
        encoding["labels"] = labels
        return dict(encoding)

    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        _SentenceDataset(examples),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    epoch_losses: list[float] = []
    for _ in range(config.epochs):
        total, batches = 0.0, 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        epoch_losses.append(total / max(batches, 1))

    model.eval()
    detector = NeuralDetector(
        model,
        tokenizer,
        checkpoint=config.checkpoint,
        device=str(device),
        max_length=config.max_length,
    )
    saved_to: str | None = None
    if config.output_dir is not None:
        detector.save(config.output_dir)
        saved_to = str(config.output_dir)
    return detector, TrainReport(
        epoch_losses=tuple(epoch_losses),
        n_examples=len(examples),
        device=str(device),
        saved_to=saved_to,
    )
