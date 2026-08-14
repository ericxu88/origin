"""Dataset schema and JSONL I/O tests (SPEC DS-1, DS-2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from origin_ml.datasets import (
    DatasetFormatError,
    DocLabel,
    DocumentRecord,
    GenerationInfo,
    LabeledSpan,
    SegmentLabel,
    TransformInfo,
    read_jsonl,
    write_jsonl,
)

GEN = GenerationInfo(
    model_family="alpha",
    model_name="alpha-small",
    provider="acme",
    temperature=0.7,
    prompt_id="p1",
)


def human_record(doc_id: str = "h1") -> DocumentRecord:
    return DocumentRecord(
        id=doc_id, text="A human wrote this.", label=DocLabel.HUMAN, source="test", group_id=doc_id
    )


def ai_record(doc_id: str = "a1") -> DocumentRecord:
    return DocumentRecord(
        id=doc_id,
        text="A model wrote this.",
        label=DocLabel.AI,
        source="test",
        group_id=doc_id,
        generation=GEN,
    )


def mixed_record(doc_id: str = "m1") -> DocumentRecord:
    text = "Human part here. AI part here."
    return DocumentRecord(
        id=doc_id,
        text=text,
        label=DocLabel.MIXED,
        source="test",
        group_id=doc_id,
        generation=GEN,
        spans=(
            LabeledSpan(start=0, end=16, label=SegmentLabel.HUMAN),
            LabeledSpan(start=17, end=30, label=SegmentLabel.AI),
        ),
    )


class TestValidation:
    def test_valid_records(self) -> None:
        human_record()
        ai_record()
        mixed_record()

    def test_mixed_requires_spans(self) -> None:
        with pytest.raises(ValueError, match="non-empty span labels"):
            DocumentRecord(
                id="m", text="x y", label=DocLabel.MIXED, source="t", group_id="m", generation=GEN
            )

    def test_mixed_requires_both_kinds_of_span(self) -> None:
        with pytest.raises(ValueError, match="both human and ai spans"):
            DocumentRecord(
                id="m",
                text="only ai here",
                label=DocLabel.MIXED,
                source="t",
                group_id="m",
                generation=GEN,
                spans=(LabeledSpan(start=0, end=12, label=SegmentLabel.AI),),
            )

    def test_spans_forbidden_on_pure_documents(self) -> None:
        with pytest.raises(ValueError, match="only valid on mixed"):
            DocumentRecord(
                id="h",
                text="pure human",
                label=DocLabel.HUMAN,
                source="t",
                group_id="h",
                spans=(LabeledSpan(start=0, end=4, label=SegmentLabel.HUMAN),),
            )

    def test_span_out_of_bounds(self) -> None:
        with pytest.raises(ValueError, match="exceeds text length"):
            DocumentRecord(
                id="m",
                text="short",
                label=DocLabel.MIXED,
                source="t",
                group_id="m",
                generation=GEN,
                spans=(
                    LabeledSpan(start=0, end=3, label=SegmentLabel.HUMAN),
                    LabeledSpan(start=3, end=99, label=SegmentLabel.AI),
                ),
            )

    def test_overlapping_spans_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-overlapping"):
            DocumentRecord(
                id="m",
                text="abcdefghij",
                label=DocLabel.MIXED,
                source="t",
                group_id="m",
                generation=GEN,
                spans=(
                    LabeledSpan(start=0, end=6, label=SegmentLabel.HUMAN),
                    LabeledSpan(start=4, end=10, label=SegmentLabel.AI),
                ),
            )

    def test_ai_requires_generation(self) -> None:
        with pytest.raises(ValueError, match="require generation metadata"):
            DocumentRecord(id="a", text="text", label=DocLabel.AI, source="t", group_id="a")

    def test_human_forbids_generation(self) -> None:
        with pytest.raises(ValueError, match="must not carry generation"):
            DocumentRecord(
                id="h",
                text="text",
                label=DocLabel.HUMAN,
                source="t",
                group_id="h",
                generation=GEN,
            )

    def test_invalid_split_rejected(self) -> None:
        payload = human_record().model_dump()
        payload["split"] = "holdout"
        with pytest.raises(ValueError, match="split must be one of"):
            DocumentRecord.model_validate(payload)

    def test_model_family_property(self) -> None:
        assert human_record().model_family is None
        assert ai_record().model_family == "alpha"

    def test_paraphrase_property(self) -> None:
        para = ai_record().model_copy(
            update={"transform": TransformInfo(kind="paraphrase", parent_id="a1")}
        )
        assert para.is_paraphrase
        assert not ai_record().is_paraphrase


class TestJsonlIO:
    def test_round_trip(self, tmp_path: Path) -> None:
        records = [human_record("h1"), ai_record("a1"), mixed_record("m1")]
        path = tmp_path / "data.jsonl"
        assert write_jsonl(path, records) == 3
        loaded = read_jsonl(path)
        assert loaded == records

    def test_error_includes_line_number(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.jsonl"
        good = human_record().model_dump_json(exclude_none=True)
        path.write_text(good + "\n" + '{"id": "x"}' + "\n", encoding="utf-8")
        with pytest.raises(DatasetFormatError, match=r"bad\.jsonl:2"):
            read_jsonl(path)

    def test_invalid_json_reported(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.jsonl"
        path.write_text("not json\n", encoding="utf-8")
        with pytest.raises(DatasetFormatError, match="invalid JSON"):
            read_jsonl(path)

    def test_duplicate_ids_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "dup.jsonl"
        write_jsonl(path, [human_record("same"), human_record("same")])
        with pytest.raises(DatasetFormatError, match="duplicate document id"):
            read_jsonl(path)

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetFormatError, match="not found"):
            read_jsonl(tmp_path / "nope.jsonl")
