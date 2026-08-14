"""Dataset pipeline: schema, validated I/O, leakage-safe splits, builders, adapters."""

from origin_ml.datasets.adapters import (
    DatasetAdapter,
    JsonlFieldMapAdapter,
    PlainTextFolderAdapter,
)
from origin_ml.datasets.builders import (
    build_mixed_document,
    make_paraphrase_record,
    paraphrase_text,
)
from origin_ml.datasets.io import DatasetFormatError, read_jsonl, read_jsonl_lazy, write_jsonl
from origin_ml.datasets.schema import (
    SPLITS,
    DocLabel,
    DocumentRecord,
    GenerationInfo,
    LabeledSpan,
    SegmentLabel,
    TransformInfo,
)
from origin_ml.datasets.splits import (
    assert_no_group_leakage,
    assign_splits,
    family_holdout_split,
    group_unit_hash,
)

__all__ = [
    "SPLITS",
    "DatasetAdapter",
    "DatasetFormatError",
    "DocLabel",
    "DocumentRecord",
    "GenerationInfo",
    "JsonlFieldMapAdapter",
    "LabeledSpan",
    "PlainTextFolderAdapter",
    "SegmentLabel",
    "TransformInfo",
    "assert_no_group_leakage",
    "assign_splits",
    "build_mixed_document",
    "family_holdout_split",
    "group_unit_hash",
    "make_paraphrase_record",
    "paraphrase_text",
    "read_jsonl",
    "read_jsonl_lazy",
    "write_jsonl",
]
