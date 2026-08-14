"""Dataset schema (SPEC DS-1): typed, validated document records.

A :class:`DocumentRecord` is the unit of every Origin dataset. Records are
stored as JSONL (one record per line, see :mod:`origin_ml.datasets.io`) and
validated with pydantic so malformed data fails loudly with helpful errors
(SPEC DS-2).

Leakage model (SPEC DS-3): every record carries a ``group_id``. Records derived
from the same underlying text (a document and its paraphrase, chunks of the
same source) share a ``group_id``, and splitters guarantee a group never
straddles train/val/test.
"""

from __future__ import annotations

from enum import StrEnum
from itertools import pairwise

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "SPLITS",
    "DocLabel",
    "DocumentRecord",
    "GenerationInfo",
    "LabeledSpan",
    "SegmentLabel",
    "TransformInfo",
]

SPLITS = ("train", "val", "test")


class DocLabel(StrEnum):
    """Document-level ground-truth label."""

    HUMAN = "human"
    AI = "ai"
    MIXED = "mixed"


class SegmentLabel(StrEnum):
    """Ground-truth label of a span inside a mixed document."""

    HUMAN = "human"
    AI = "ai"


class LabeledSpan(BaseModel):
    """A labeled character span ``[start, end)`` inside a document."""

    model_config = ConfigDict(frozen=True)

    start: int = Field(ge=0)
    end: int
    label: SegmentLabel

    @model_validator(mode="after")
    def _check_bounds(self) -> LabeledSpan:
        if self.end <= self.start:
            raise ValueError(f"span end ({self.end}) must be > start ({self.start})")
        return self


class GenerationInfo(BaseModel):
    """Provenance of machine-generated text (SPEC DS-1)."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    model_family: str = Field(min_length=1, description="e.g. 'gpt', 'claude', 'llama'")
    model_name: str = Field(min_length=1, description="e.g. 'gpt-4o-mini'")
    provider: str | None = None
    temperature: float | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, ge=0, le=1)
    prompt_id: str | None = None
    prompt_summary: str | None = None


class TransformInfo(BaseModel):
    """Lineage for derived texts (paraphrase, human edit, ...)."""

    model_config = ConfigDict(frozen=True)

    kind: str = Field(min_length=1, description="e.g. 'paraphrase', 'edit'")
    parent_id: str = Field(min_length=1)
    tool: str | None = None


class DocumentRecord(BaseModel):
    """One dataset document with full metadata (SPEC DS-1)."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    label: DocLabel
    spans: tuple[LabeledSpan, ...] = ()
    source: str = Field(
        min_length=1, description="provenance, e.g. 'gutenberg', 'synthetic-fixture'"
    )
    group_id: str = Field(min_length=1, description="leakage group; see module docstring")
    generation: GenerationInfo | None = None
    transform: TransformInfo | None = None
    split: str | None = None
    meta: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate(self) -> DocumentRecord:
        if self.label is DocLabel.MIXED:
            if not self.spans:
                raise ValueError("mixed documents require non-empty span labels")
            if not any(s.label is SegmentLabel.AI for s in self.spans) or not any(
                s.label is SegmentLabel.HUMAN for s in self.spans
            ):
                raise ValueError("mixed documents need both human and ai spans")
        elif self.spans:
            raise ValueError(f"span labels are only valid on mixed documents (label={self.label})")

        for span in self.spans:
            if span.end > len(self.text):
                raise ValueError(
                    f"span ({span.start}, {span.end}) exceeds text length {len(self.text)}"
                )
        for a, b in pairwise(self.spans):
            if b.start < a.end:
                raise ValueError(f"spans must be sorted and non-overlapping: {a} then {b}")

        if self.label in (DocLabel.AI, DocLabel.MIXED) and self.generation is None:
            raise ValueError(f"{self.label} documents require generation metadata")
        if self.label is DocLabel.HUMAN and self.generation is not None:
            raise ValueError("human documents must not carry generation metadata")

        if self.split is not None and self.split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS} or None, got {self.split!r}")
        return self

    @property
    def model_family(self) -> str | None:
        """Generating model family, or None for purely human documents."""
        return self.generation.model_family if self.generation is not None else None

    @property
    def is_paraphrase(self) -> bool:
        return self.transform is not None and self.transform.kind == "paraphrase"
