"""Record-level training entry point for the baseline detector (SPEC C-2)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from origin_ml.datasets.schema import DocLabel, DocumentRecord
from origin_ml.detectors.classical import BaselineConfig, BaselineDetector
from origin_ml.features.pipeline import FeaturePipeline

__all__ = ["extract_feature_matrix", "train_baseline"]


def extract_feature_matrix(
    records: Sequence[DocumentRecord], pipeline: FeaturePipeline
) -> np.ndarray:
    """Feature matrix (one row per record) in pipeline feature order."""
    if not records:
        return np.empty((0, len(pipeline.feature_names)), dtype=np.float64)
    return np.asarray(
        [pipeline.extract(record.text).values for record in records], dtype=np.float64
    )


def train_baseline(
    records: Sequence[DocumentRecord],
    pipeline: FeaturePipeline,
    *,
    config: BaselineConfig | None = None,
    split: str | None = "train",
    dataset_name: str | None = None,
) -> BaselineDetector:
    """Train the baseline on pure human/AI documents of ``split``.

    Mixed documents are excluded: the document-level baseline is a binary
    human-vs-ai classifier; mixed handling composes sentence-level evidence
    (SPEC L-3) rather than training on ambiguous whole-document labels.
    """
    usable = [
        r
        for r in records
        if r.label in (DocLabel.HUMAN, DocLabel.AI) and (split is None or r.split == split)
    ]
    if not usable:
        raise ValueError("no pure human/ai records in the requested split")

    features = extract_feature_matrix(usable, pipeline)
    labels = np.asarray([1 if r.label is DocLabel.AI else 0 for r in usable], dtype=np.int64)
    scorer = pipeline.scorer
    return BaselineDetector.train(
        features,
        labels,
        pipeline.feature_names,
        config=config,
        scorer_name=scorer.name if scorer is not None else None,
        training_meta={
            "dataset": dataset_name,
            "split": split,
            "n_human": int((labels == 0).sum()),
            "n_ai": int((labels == 1).sum()),
        },
    )
