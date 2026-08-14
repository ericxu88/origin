"""Sentence-level training example derivation, shared by classical and neural
training (torch-free module).

- human documents contribute their sentences with label 0,
- ai documents contribute their sentences with label 1,
- mixed documents contribute per-sentence labels from ground-truth spans
  (sentence midpoint inside an AI span → label 1).
"""

from __future__ import annotations

from origin_ml.datasets.schema import DocLabel, DocumentRecord, SegmentLabel
from origin_ml.text.segmentation import segment_sentences

__all__ = ["sentence_examples"]


def sentence_examples(records: list[DocumentRecord]) -> list[tuple[str, int]]:
    """Flatten records into ``(sentence_text, label)`` pairs."""
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
