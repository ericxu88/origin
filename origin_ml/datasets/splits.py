"""Leakage-safe dataset splitting (SPEC DS-3, DS-4).

All split assignment is a pure function of ``(seed, group_id)`` via a stable
BLAKE2 hash, so:

- the same corpus always splits identically for a given seed, and
- every record of a group lands in the same split by construction — a
  paraphrase can never leak across the boundary from its parent.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence

from origin_ml.datasets.schema import DocLabel, DocumentRecord

__all__ = [
    "assert_no_group_leakage",
    "assign_splits",
    "family_holdout_split",
    "group_unit_hash",
]


def group_unit_hash(seed: int, group_id: str) -> float:
    """Deterministic hash of a group into ``[0, 1)``, stable across platforms."""
    digest = hashlib.blake2b(f"{seed}:{group_id}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(2**64)


def assign_splits(
    records: Sequence[DocumentRecord],
    *,
    train: float = 0.8,
    val: float = 0.0,
    test: float = 0.2,
    seed: int = 0,
) -> list[DocumentRecord]:
    """Assign group-aware train/val/test splits (SPEC DS-3).

    Ratios must sum to 1. Returns copies with ``split`` set; input order is
    preserved.
    """
    total = train + val + test
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"split ratios must sum to 1, got {total}")
    if min(train, val, test) < 0:
        raise ValueError("split ratios must be non-negative")

    out: list[DocumentRecord] = []
    for record in records:
        u = group_unit_hash(seed, record.group_id)
        if u < train:
            split = "train"
        elif u < train + val:
            split = "val"
        else:
            split = "test"
        out.append(record.model_copy(update={"split": split}))
    return out


def family_holdout_split(
    records: Sequence[DocumentRecord],
    holdout_families: Iterable[str],
    *,
    human_test_fraction: float = 0.3,
    seed: int = 0,
) -> tuple[list[DocumentRecord], list[DocumentRecord]]:
    """Split for unseen-model-family evaluation (SPEC DS-4).

    Every AI/mixed document whose generating ``model_family`` is in
    ``holdout_families`` goes to test; other AI/mixed documents go to train.
    Human documents are hashed group-wise into both sides so each side keeps
    human negatives. Returns ``(train, test)`` with ``split`` set.
    """
    holdout = set(holdout_families)
    if not 0.0 < human_test_fraction < 1.0:
        raise ValueError("human_test_fraction must be in (0, 1)")

    train_out: list[DocumentRecord] = []
    test_out: list[DocumentRecord] = []
    for record in records:
        if record.label is DocLabel.HUMAN:
            to_test = group_unit_hash(seed, record.group_id) < human_test_fraction
        else:
            family = record.model_family
            assert family is not None  # schema guarantees generation on ai/mixed
            to_test = family in holdout
        if to_test:
            test_out.append(record.model_copy(update={"split": "test"}))
        else:
            train_out.append(record.model_copy(update={"split": "train"}))
    return train_out, test_out


def assert_no_group_leakage(records: Sequence[DocumentRecord]) -> None:
    """Raise ``ValueError`` if any group appears in more than one split."""
    seen: dict[str, set[str]] = {}
    for record in records:
        if record.split is not None:
            seen.setdefault(record.group_id, set()).add(record.split)
    leaks = {group: splits for group, splits in seen.items() if len(splits) > 1}
    if leaks:
        detail = ", ".join(f"{g} -> {sorted(s)}" for g, s in sorted(leaks.items()))
        raise ValueError(f"group leakage across splits: {detail}")
