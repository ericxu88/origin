"""Dataset builders: mixed documents and deterministic paraphrases (DS-6, DS-7).

The mixed-document builder splices sentences from a human text and an AI text
into one document with exact character span labels, used both for the sample
dataset and for localization evaluation ground truth.

The paraphraser is a deterministic, rule-based lexical paraphraser. It is a
*fixture-quality* transform for robustness evaluation without API access;
external paraphrase tools can be plugged in through the same
:class:`~origin_ml.datasets.schema.TransformInfo` lineage.
"""

from __future__ import annotations

import random
import re

from origin_ml.datasets.schema import (
    DocLabel,
    DocumentRecord,
    GenerationInfo,
    LabeledSpan,
    SegmentLabel,
    TransformInfo,
)
from origin_ml.text.segmentation import segment_sentences

__all__ = ["build_mixed_document", "make_paraphrase_record", "paraphrase_text"]


def build_mixed_document(
    *,
    doc_id: str,
    human_text: str,
    ai_text: str,
    generation: GenerationInfo,
    group_id: str,
    source: str,
    seed: int = 0,
    separator: str = " ",
) -> DocumentRecord:
    """Interleave sentences of ``human_text`` and ``ai_text`` with span labels.

    Alternates between the two sources in seeded random block sizes of 1-3
    sentences until both are exhausted. Adjacent same-label sentences are
    merged into one span; the single separator character between spans of
    different labels stays unlabeled.

    Leakage note (SPEC DS-3): the caller chooses ``group_id``. Parent texts
    used for mixing must either share that group or be reserved exclusively
    for mixing (the sample-data generator does the latter).
    """
    human_sentences = [s.text for s in segment_sentences(human_text)]
    ai_sentences = [s.text for s in segment_sentences(ai_text)]
    if not human_sentences or not ai_sentences:
        raise ValueError("both source texts must contain at least one sentence")

    rng = random.Random(seed)
    queue = {SegmentLabel.HUMAN: human_sentences, SegmentLabel.AI: ai_sentences}
    current = rng.choice((SegmentLabel.HUMAN, SegmentLabel.AI))

    parts: list[str] = []
    spans: list[LabeledSpan] = []
    cursor = 0
    while queue[SegmentLabel.HUMAN] or queue[SegmentLabel.AI]:
        if not queue[current]:
            current = SegmentLabel.AI if current is SegmentLabel.HUMAN else SegmentLabel.HUMAN
        take = min(rng.randint(1, 3), len(queue[current]))
        segment = separator.join(queue[current][:take])
        del queue[current][:take]

        if parts:
            cursor += len(separator)
        start = cursor
        parts.append(segment)
        cursor += len(segment)

        if spans and spans[-1].label is current:
            spans[-1] = LabeledSpan(start=spans[-1].start, end=cursor, label=current)
        else:
            spans.append(LabeledSpan(start=start, end=cursor, label=current))
        current = SegmentLabel.AI if current is SegmentLabel.HUMAN else SegmentLabel.HUMAN

    return DocumentRecord(
        id=doc_id,
        text=separator.join(parts),
        label=DocLabel.MIXED,
        spans=tuple(spans),
        source=source,
        group_id=group_id,
        generation=generation,
    )


# Deterministic lexical substitutions used by the fixture paraphraser.
_SYNONYMS: dict[str, str] = {
    "big": "large",
    "small": "little",
    "quick": "fast",
    "quickly": "rapidly",
    "begin": "start",
    "began": "started",
    "end": "finish",
    "important": "significant",
    "show": "reveal",
    "shows": "reveals",
    "showed": "revealed",
    "use": "employ",
    "used": "employed",
    "help": "assist",
    "helps": "assists",
    "many": "numerous",
    "often": "frequently",
    "almost": "nearly",
    "however": "nevertheless",
    "because": "since",
    "beautiful": "lovely",
    "difficult": "hard",
    "easy": "simple",
    "very": "quite",
    "old": "aged",
    "said": "stated",
    "walked": "strolled",
    "looked": "gazed",
    "house": "dwelling",
    "night": "evening",
}


def paraphrase_text(text: str, seed: int = 0) -> str:
    """Deterministic rule-based paraphrase: seeded synonym substitution.

    Roughly half of the substitutable words (chosen by seeded RNG) are
    replaced, preserving leading capitalization. Sentence structure and
    offsets change, content stays recognizable — by design this simulates a
    light paraphrase attack, not a strong one.
    """
    rng = random.Random(seed)

    def replace(match: re.Match[str]) -> str:
        word = match.group()
        replacement = _SYNONYMS.get(word.lower())
        if replacement is None or rng.random() >= 0.5:
            return word
        if word[0].isupper():
            replacement = replacement[0].upper() + replacement[1:]
        return replacement

    return re.sub(r"[A-Za-z']+", replace, text)


def make_paraphrase_record(parent: DocumentRecord, *, doc_id: str, seed: int = 0) -> DocumentRecord:
    """Create the paraphrased sibling of ``parent`` with correct lineage.

    The child keeps the parent's label, generation metadata, and — critically
    for leakage safety — its ``group_id`` (SPEC DS-3), and records
    ``TransformInfo(kind="paraphrase")``. Mixed parents are not supported
    because paraphrasing invalidates their span offsets.
    """
    if parent.label is DocLabel.MIXED:
        raise ValueError("cannot paraphrase mixed documents: span labels would be invalidated")
    return DocumentRecord(
        id=doc_id,
        text=paraphrase_text(parent.text, seed=seed),
        label=parent.label,
        source=parent.source,
        group_id=parent.group_id,
        generation=parent.generation,
        transform=TransformInfo(kind="paraphrase", parent_id=parent.id, tool="origin-lexical"),
        meta=dict(parent.meta),
    )
