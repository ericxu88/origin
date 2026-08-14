"""Adapter tests (SPEC DS-6)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from origin_ml.datasets import (
    DatasetAdapter,
    DocLabel,
    JsonlFieldMapAdapter,
    PlainTextFolderAdapter,
)


class TestPlainTextFolderAdapter:
    def test_imports_txt_files(self, tmp_path: Path) -> None:
        (tmp_path / "essay1.txt").write_text("A human essay.", encoding="utf-8")
        (tmp_path / "essay2.txt").write_text("Another essay.", encoding="utf-8")
        (tmp_path / "notes.md").write_text("ignored", encoding="utf-8")
        adapter = PlainTextFolderAdapter(
            tmp_path, label=DocLabel.HUMAN, source="folder-corpus", id_prefix="fc-"
        )
        assert isinstance(adapter, DatasetAdapter)
        records = list(adapter.records())
        assert [r.id for r in records] == ["fc-essay1", "fc-essay2"]
        assert all(r.label is DocLabel.HUMAN for r in records)
        assert all(r.source == "folder-corpus" for r in records)

    def test_skips_empty_files(self, tmp_path: Path) -> None:
        (tmp_path / "empty.txt").write_text("   \n", encoding="utf-8")
        adapter = PlainTextFolderAdapter(tmp_path, label=DocLabel.HUMAN, source="s")
        assert list(adapter.records()) == []

    def test_rejects_missing_folder(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not a directory"):
            PlainTextFolderAdapter(tmp_path / "nope", label=DocLabel.HUMAN, source="s")


class TestJsonlFieldMapAdapter:
    def test_maps_fields(self, tmp_path: Path) -> None:
        path = tmp_path / "corpus.jsonl"
        rows = [
            {"body": "Model output one.", "model": "gpt-4o", "temp": 0.3, "kind": "machine"},
            {"body": "A person wrote this.", "kind": "person"},
        ]
        path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        adapter = JsonlFieldMapAdapter(
            path,
            text_field="body",
            source="public-corpus",
            label_field="kind",
            label_map={"machine": DocLabel.AI, "person": DocLabel.HUMAN},
            model_family_field="model",
            model_name_field="model",
            temperature_field="temp",
        )
        records = list(adapter.records())
        assert [r.label for r in records] == [DocLabel.AI, DocLabel.HUMAN]
        ai = records[0]
        assert ai.generation is not None
        assert ai.generation.model_name == "gpt-4o"
        assert ai.generation.temperature == 0.3
        assert records[1].generation is None

    def test_fixed_label_mode(self, tmp_path: Path) -> None:
        path = tmp_path / "ai.jsonl"
        path.write_text(json.dumps({"body": "Generated text."}), encoding="utf-8")
        adapter = JsonlFieldMapAdapter(
            path, text_field="body", source="s", label=DocLabel.AI, model_name_field="missing"
        )
        (record,) = list(adapter.records())
        assert record.generation is not None
        assert record.generation.model_family == "unknown"

    def test_unmapped_label_reported_with_line(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.jsonl"
        path.write_text(json.dumps({"body": "x y", "kind": "alien"}), encoding="utf-8")
        adapter = JsonlFieldMapAdapter(
            path,
            text_field="body",
            source="s",
            label_field="kind",
            label_map={"person": DocLabel.HUMAN},
        )
        with pytest.raises(ValueError, match=r"bad\.jsonl:1: unmapped label"):
            list(adapter.records())

    def test_config_validation(self, tmp_path: Path) -> None:
        path = tmp_path / "x.jsonl"
        with pytest.raises(ValueError, match="exactly one of label or label_field"):
            JsonlFieldMapAdapter(path, text_field="t", source="s")
        with pytest.raises(ValueError, match="requires label_map"):
            JsonlFieldMapAdapter(path, text_field="t", source="s", label_field="kind")
