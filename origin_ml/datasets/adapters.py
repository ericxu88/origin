"""Adapters for importing external data into the Origin schema (SPEC DS-6).

Adapters turn external corpora into validated :class:`DocumentRecord` streams:

- :class:`PlainTextFolderAdapter` — a directory of ``.txt`` files (the common
  shape of public human corpora such as Project Gutenberg extracts).
- :class:`JsonlFieldMapAdapter` — arbitrary JSONL corpora via a declarative
  field mapping (the common shape of published AI-text datasets).

No adapter needs an API key. Scripts that *generate* new AI samples against a
provider belong in ``scripts/`` and read credentials from the environment;
they are never imported by tests.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol, runtime_checkable

from origin_ml.datasets.schema import DocLabel, DocumentRecord, GenerationInfo

__all__ = ["DatasetAdapter", "JsonlFieldMapAdapter", "PlainTextFolderAdapter"]


@runtime_checkable
class DatasetAdapter(Protocol):
    """Yields validated document records from some external source."""

    def records(self) -> Iterator[DocumentRecord]: ...


class PlainTextFolderAdapter:
    """Each ``*.txt`` file in a folder becomes one document.

    ``group_id`` defaults to the file stem; pass ``group_prefix`` to namespace
    groups when merging multiple corpora.
    """

    def __init__(
        self,
        folder: Path,
        *,
        label: DocLabel,
        source: str,
        generation: GenerationInfo | None = None,
        id_prefix: str = "",
        group_prefix: str = "",
    ) -> None:
        if not folder.is_dir():
            raise ValueError(f"not a directory: {folder}")
        self._folder = folder
        self._label = label
        self._source = source
        self._generation = generation
        self._id_prefix = id_prefix
        self._group_prefix = group_prefix

    def records(self) -> Iterator[DocumentRecord]:
        for path in sorted(self._folder.glob("*.txt")):
            text = path.read_text(encoding="utf-8").strip()
            if not text:
                continue
            yield DocumentRecord(
                id=f"{self._id_prefix}{path.stem}",
                text=text,
                label=self._label,
                source=self._source,
                group_id=f"{self._group_prefix}{path.stem}",
                generation=self._generation,
            )


class JsonlFieldMapAdapter:
    """Map arbitrary JSONL rows onto the Origin schema.

    Example — a public detection corpus with ``{"text": ..., "model": ...}``::

        JsonlFieldMapAdapter(
            path,
            text_field="text",
            label=DocLabel.AI,
            source="some-public-corpus",
            model_name_field="model",
            model_family_field="model",
        )
    """

    def __init__(
        self,
        path: Path,
        *,
        text_field: str,
        source: str,
        label: DocLabel | None = None,
        label_field: str | None = None,
        label_map: dict[str, DocLabel] | None = None,
        id_field: str | None = None,
        group_field: str | None = None,
        model_family_field: str | None = None,
        model_name_field: str | None = None,
        temperature_field: str | None = None,
        id_prefix: str = "",
    ) -> None:
        if (label is None) == (label_field is None):
            raise ValueError("provide exactly one of label or label_field")
        if label_field is not None and label_map is None:
            raise ValueError("label_field requires label_map")
        self._path = path
        self._text_field = text_field
        self._source = source
        self._label = label
        self._label_field = label_field
        self._label_map = label_map or {}
        self._id_field = id_field
        self._group_field = group_field
        self._model_family_field = model_family_field
        self._model_name_field = model_name_field
        self._temperature_field = temperature_field
        self._id_prefix = id_prefix

    def _label_for(self, row: dict[str, object], line_no: int) -> DocLabel:
        if self._label is not None:
            return self._label
        assert self._label_field is not None
        raw = str(row.get(self._label_field, ""))
        if raw not in self._label_map:
            raise ValueError(f"{self._path}:{line_no}: unmapped label value {raw!r}")
        return self._label_map[raw]

    def _generation_for(self, row: dict[str, object], label: DocLabel) -> GenerationInfo | None:
        if label is DocLabel.HUMAN:
            return None
        family = str(row.get(self._model_family_field, "")) if self._model_family_field else ""
        name = str(row.get(self._model_name_field, "")) if self._model_name_field else ""
        temperature: float | None = None
        if self._temperature_field is not None and row.get(self._temperature_field) is not None:
            temperature = float(str(row[self._temperature_field]))
        return GenerationInfo(
            model_family=family or "unknown",
            model_name=name or "unknown",
            temperature=temperature,
        )

    def records(self) -> Iterator[DocumentRecord]:
        with self._path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row: dict[str, object] = json.loads(line)
                text = str(row.get(self._text_field, "")).strip()
                if not text:
                    continue
                label = self._label_for(row, line_no)
                doc_id = (
                    f"{self._id_prefix}{row[self._id_field]}"
                    if self._id_field is not None and self._id_field in row
                    else f"{self._id_prefix}{self._path.stem}-{line_no}"
                )
                group = (
                    str(row[self._group_field])
                    if self._group_field is not None and self._group_field in row
                    else doc_id
                )
                yield DocumentRecord(
                    id=doc_id,
                    text=text,
                    label=label,
                    source=self._source,
                    group_id=group,
                    generation=self._generation_for(row, label),
                )
