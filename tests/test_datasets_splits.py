"""Leakage-safe split tests (SPEC DS-3, DS-4)."""

from __future__ import annotations

import random

import pytest

from origin_ml.datasets import (
    DocLabel,
    DocumentRecord,
    GenerationInfo,
    assert_no_group_leakage,
    assign_splits,
    family_holdout_split,
)


def make_record(
    doc_id: str, group: str, label: DocLabel, family: str | None = None
) -> DocumentRecord:
    generation = (
        GenerationInfo(model_family=family or "alpha", model_name=f"{family or 'alpha'}-1")
        if label is not DocLabel.HUMAN
        else None
    )
    return DocumentRecord(
        id=doc_id,
        text=f"Document {doc_id} body text.",
        label=label,
        source="test",
        group_id=group,
        generation=generation,
    )


def synthetic_corpus(n_groups: int = 60, seed: int = 3) -> list[DocumentRecord]:
    """Groups of 1-3 sibling records (parent + paraphrases) across labels/families."""
    rng = random.Random(seed)
    records: list[DocumentRecord] = []
    families = ["alpha", "beta", "gamma"]
    for g in range(n_groups):
        label = rng.choice([DocLabel.HUMAN, DocLabel.AI])
        family = rng.choice(families) if label is DocLabel.AI else None
        group = f"group-{g:03d}"
        for sibling in range(rng.randint(1, 3)):
            records.append(make_record(f"doc-{g:03d}-{sibling}", group, label, family))
    return records


class TestAssignSplits:
    def test_deterministic(self) -> None:
        corpus = synthetic_corpus()
        a = assign_splits(corpus, seed=1)
        b = assign_splits(corpus, seed=1)
        assert [r.split for r in a] == [r.split for r in b]

    def test_seed_changes_assignment(self) -> None:
        corpus = synthetic_corpus()
        a = assign_splits(corpus, seed=1)
        b = assign_splits(corpus, seed=2)
        assert [r.split for r in a] != [r.split for r in b]

    def test_groups_never_straddle_splits(self) -> None:
        """Core leakage property (DS-3): all siblings share a split."""
        for seed in range(10):
            split_records = assign_splits(
                synthetic_corpus(), train=0.6, val=0.2, test=0.2, seed=seed
            )
            assert_no_group_leakage(split_records)

    def test_ratios_roughly_respected(self) -> None:
        singles = [make_record(f"d{i}", f"g{i}", DocLabel.HUMAN) for i in range(1000)]
        split_records = assign_splits(singles, train=0.8, val=0.0, test=0.2, seed=0)
        train_frac = sum(1 for r in split_records if r.split == "train") / len(split_records)
        assert 0.75 < train_frac < 0.85

    def test_bad_ratios_rejected(self) -> None:
        with pytest.raises(ValueError, match="sum to 1"):
            assign_splits([], train=0.5, val=0.0, test=0.2)


class TestFamilyHoldout:
    def test_holdout_family_never_in_train(self) -> None:
        corpus = synthetic_corpus(n_groups=120)
        train, test = family_holdout_split(corpus, holdout_families={"gamma"})
        assert all(r.model_family != "gamma" for r in train)
        assert all(r.model_family in (None, "gamma") for r in test)

    def test_humans_present_on_both_sides(self) -> None:
        corpus = synthetic_corpus(n_groups=120)
        train, test = family_holdout_split(corpus, holdout_families={"gamma"})
        assert any(r.label is DocLabel.HUMAN for r in train)
        assert any(r.label is DocLabel.HUMAN for r in test)

    def test_human_groups_do_not_straddle(self) -> None:
        corpus = synthetic_corpus(n_groups=120)
        train, test = family_holdout_split(corpus, holdout_families={"beta"})
        assert_no_group_leakage([*train, *test])

    def test_bad_fraction_rejected(self) -> None:
        with pytest.raises(ValueError, match="human_test_fraction"):
            family_holdout_split([], {"alpha"}, human_test_fraction=1.5)


class TestLeakageGuard:
    def test_detects_leak(self) -> None:
        a = make_record("d1", "shared", DocLabel.HUMAN).model_copy(update={"split": "train"})
        b = make_record("d2", "shared", DocLabel.HUMAN).model_copy(update={"split": "test"})
        with pytest.raises(ValueError, match="group leakage"):
            assert_no_group_leakage([a, b])

    def test_accepts_clean_assignment(self) -> None:
        a = make_record("d1", "g1", DocLabel.HUMAN).model_copy(update={"split": "train"})
        b = make_record("d2", "g1", DocLabel.HUMAN).model_copy(update={"split": "train"})
        assert_no_group_leakage([a, b])
