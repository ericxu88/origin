"""Mixed-document builder and paraphraser tests (SPEC DS-6, DS-7)."""

from __future__ import annotations

from itertools import pairwise

import pytest

from origin_ml.datasets import (
    DocLabel,
    DocumentRecord,
    GenerationInfo,
    SegmentLabel,
    build_mixed_document,
    make_paraphrase_record,
    paraphrase_text,
)

GEN = GenerationInfo(model_family="alpha", model_name="alpha-small", temperature=0.7)

HUMAN = "The crooked lantern flickered. Nobody slept that night. Rain kept falling until dawn."
AI = (
    "Additionally, the system provides clear benefits. Moreover, the process supports "
    "better outcomes. Furthermore, the approach ensures consistent results."
)


def build(seed: int = 0) -> DocumentRecord:
    return build_mixed_document(
        doc_id="mix-1",
        human_text=HUMAN,
        ai_text=AI,
        generation=GEN,
        group_id="mix-1",
        source="test",
        seed=seed,
    )


class TestMixedBuilder:
    def test_produces_valid_mixed_record(self) -> None:
        record = build()
        assert record.label is DocLabel.MIXED
        assert record.spans
        assert record.generation is GEN

    def test_spans_are_sorted_disjoint_and_alternating(self) -> None:
        record = build()
        for a, b in pairwise(record.spans):
            assert a.end <= b.start
            assert a.label is not b.label  # merged: adjacent spans always differ

    def test_spans_cover_only_source_sentences(self) -> None:
        """Every span's text consists of sentences from its own source."""
        record = build()
        for span in record.spans:
            fragment = record.text[span.start : span.end]
            source = HUMAN if span.label is SegmentLabel.HUMAN else AI
            for sentence in fragment.split(". "):
                cleaned = sentence.strip()
                assert cleaned and cleaned.rstrip(".") in source

    def test_all_sentences_used_exactly_once(self) -> None:
        record = build()
        rebuilt_lengths = sum(span.end - span.start for span in record.spans)
        gaps = len(record.spans) - 1
        assert rebuilt_lengths + gaps == len(record.text)

    def test_deterministic_per_seed(self) -> None:
        assert build(seed=5) == build(seed=5)
        assert build(seed=5).text != build(seed=6).text or (
            build(seed=5).spans != build(seed=6).spans
        )

    def test_rejects_empty_sources(self) -> None:
        with pytest.raises(ValueError, match="at least one sentence"):
            build_mixed_document(
                doc_id="m",
                human_text="   ",
                ai_text=AI,
                generation=GEN,
                group_id="m",
                source="test",
            )


class TestParaphraser:
    def test_deterministic(self) -> None:
        text = "The big house was very beautiful because many people used it often."
        assert paraphrase_text(text, seed=1) == paraphrase_text(text, seed=1)

    def test_changes_text_but_preserves_capitalization(self) -> None:
        text = "Big changes happen. Very often the old house looked beautiful."
        result = paraphrase_text(text, seed=0)
        assert result != text
        assert result.split(". ")[0][0].isupper()

    def test_paraphrase_record_keeps_group_and_lineage(self) -> None:
        parent = DocumentRecord(
            id="a1",
            text="The model said it was very important to use the big system often.",
            label=DocLabel.AI,
            source="test",
            group_id="family-group",
            generation=GEN,
        )
        child = make_paraphrase_record(parent, doc_id="a1-para", seed=2)
        assert child.group_id == parent.group_id  # leakage-safety (DS-3)
        assert child.label is parent.label
        assert child.transform is not None
        assert child.transform.kind == "paraphrase"
        assert child.transform.parent_id == parent.id
        assert child.is_paraphrase

    def test_mixed_parent_rejected(self) -> None:
        mixed = build()
        with pytest.raises(ValueError, match="cannot paraphrase mixed"):
            make_paraphrase_record(mixed, doc_id="x")
