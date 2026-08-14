"""Validated JSONL reading/writing for datasets (SPEC DS-2)."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from pydantic import ValidationError

from origin_ml.datasets.schema import DocumentRecord

__all__ = ["DatasetFormatError", "read_jsonl", "read_jsonl_lazy", "write_jsonl"]


class DatasetFormatError(ValueError):
    """A dataset file failed validation; message pinpoints file and line."""


def read_jsonl_lazy(path: Path) -> Iterator[DocumentRecord]:
    """Yield validated records; raise :class:`DatasetFormatError` with context."""
    if not path.exists():
        raise DatasetFormatError(f"dataset file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetFormatError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            try:
                yield DocumentRecord.model_validate(payload)
            except ValidationError as exc:
                raise DatasetFormatError(f"{path}:{line_no}: invalid record: {exc}") from exc


def read_jsonl(path: Path) -> list[DocumentRecord]:
    """Read and validate a whole JSONL dataset, checking id uniqueness."""
    records = list(read_jsonl_lazy(path))
    seen: set[str] = set()
    for record in records:
        if record.id in seen:
            raise DatasetFormatError(f"{path}: duplicate document id {record.id!r}")
        seen.add(record.id)
    return records


def write_jsonl(path: Path, records: Iterable[DocumentRecord]) -> int:
    """Write records as JSONL (UTF-8, one compact object per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json(exclude_none=True, exclude_defaults=True))
            handle.write("\n")
            count += 1
    return count
