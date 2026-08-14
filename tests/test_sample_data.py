"""Committed sample dataset stays valid and complete (SPEC DS-5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from origin_ml.datasets import (
    DocLabel,
    DocumentRecord,
    assert_no_group_leakage,
    assign_splits,
    family_holdout_split,
    read_jsonl,
)

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample" / "documents.jsonl"


@pytest.fixture(scope="module")
def records() -> list[DocumentRecord]:
    return read_jsonl(SAMPLE)


class TestSampleData:
    def test_exists_and_validates(self, records: list[DocumentRecord]) -> None:
        assert len(records) >= 80

    def test_contains_all_classes(self, records: list[DocumentRecord]) -> None:
        labels = {r.label for r in records}
        assert labels == {DocLabel.HUMAN, DocLabel.AI, DocLabel.MIXED}
        assert any(r.is_paraphrase for r in records)

    def test_at_least_three_model_families(self, records: list[DocumentRecord]) -> None:
        families = {r.model_family for r in records if r.model_family is not None}
        assert len(families) >= 3

    def test_generation_metadata_present(self, records: list[DocumentRecord]) -> None:
        for record in records:
            if record.label in (DocLabel.AI, DocLabel.MIXED):
                assert record.generation is not None
                assert record.generation.temperature is not None

    def test_mixed_spans_round_trip(self, records: list[DocumentRecord]) -> None:
        mixed = [r for r in records if r.label is DocLabel.MIXED]
        assert mixed
        for record in mixed:
            for span in record.spans:
                assert record.text[span.start : span.end].strip()

    def test_paraphrases_share_parent_group(self, records: list[DocumentRecord]) -> None:
        by_id = {r.id: r for r in records}
        paraphrases = [r for r in records if r.is_paraphrase]
        assert paraphrases
        for para in paraphrases:
            assert para.transform is not None
            parent = by_id[para.transform.parent_id]
            assert para.group_id == parent.group_id

    def test_splits_leakage_free(self, records: list[DocumentRecord]) -> None:
        assert_no_group_leakage(assign_splits(records, seed=0))

    def test_family_holdout_works(self, records: list[DocumentRecord]) -> None:
        train, test = family_holdout_split(records, holdout_families={"gamma"})
        assert train and test
        assert all(r.model_family != "gamma" for r in train)
