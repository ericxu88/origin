"""Robustness evaluation: seen families, per-family breakdown, and GPT-4 OOD.

Measures how well trained classical artifacts generalize to newer / unseen
model families (SPEC RQ4/RQ5):

1. held-out test split of the training corpus (seen families), overall and
   broken down per model family;
2. sentence localization + mixed-doc labelling on the held-out split;
3. the MAGE **GPT-4 out-of-distribution testbed** — a model family AND domain
   never seen in training (the "newer model" proxy);
4. the paraphrase-attacked variant of that testbed.

Usage::

    uv run python scripts/eval_robust.py --artifacts artifacts/robust \
        --dataset data/combined/documents.jsonl \
        --ood data/mage/ood_gpt4.jsonl --ood-para data/mage/ood_gpt4_para.jsonl \
        --out runs/robust-metrics.json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from origin_ml.datasets import DocLabel, DocumentRecord, assign_splits, read_jsonl
from origin_ml.detectors.classical import BaselineDetector
from origin_ml.evaluation import (
    AblationSystem,
    evaluate_doc_classification,
    evaluate_localization,
    evaluate_mixed_doc_labels,
)
from origin_ml.evaluation.evaluate import default_sentence_scorer
from origin_ml.features.pipeline import build_default_pipeline
from origin_ml.scoring.hf import HFCausalScorer


def build_system(artifacts: Path, scorer_checkpoint: str) -> AblationSystem:
    pipeline = build_default_pipeline(scorer=HFCausalScorer(scorer_checkpoint))
    doc_model = BaselineDetector.load(artifacts / "doc_baseline.json")
    sent_model = BaselineDetector.load(artifacts / "sentence_baseline.json")

    def doc_p(text: str) -> float:
        return doc_model.predict_proba_one(
            np.asarray(pipeline.extract(text).values, dtype=np.float64)
        )

    def sentence_probs(sentences: list[str]) -> list[float]:
        matrix = np.asarray([pipeline.extract(s).values for s in sentences], dtype=np.float64)
        return [float(p) for p in sent_model.predict_proba(matrix)]

    return AblationSystem(
        name="robust-classical",
        description=f"artifacts from {artifacts}",
        doc_p_ai=doc_p,
        sentence_p_ai=default_sentence_scorer(sentence_probs),
    )


def doc_metrics_line(
    name: str, records: Sequence[DocumentRecord], system: AblationSystem
) -> dict[str, object]:
    metrics = evaluate_doc_classification(records, system)
    auroc = f"{metrics.auroc:.3f}" if metrics.auroc is not None else "  n/a"
    print(
        f"  {name:28s} n={metrics.n:5d}  acc={metrics.accuracy:.3f}  "
        f"f1={metrics.f1:.3f}  auroc={auroc}  brier={metrics.brier:.3f}"
    )
    return {"name": name, **metrics.model_dump()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--ood", type=Path, required=True)
    parser.add_argument("--ood-para", type=Path, required=True)
    parser.add_argument("--scorer-checkpoint", default="distilgpt2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    system = build_system(args.artifacts, args.scorer_checkpoint)
    results: dict[str, object] = {"artifacts": str(args.artifacts)}

    records = assign_splits(
        read_jsonl(args.dataset),
        train=args.train_ratio,
        val=0.0,
        test=1.0 - args.train_ratio,
        seed=args.seed,
    )
    test = [r for r in records if r.split == "test"]
    pure_test = [r for r in test if r.label in (DocLabel.HUMAN, DocLabel.AI)]
    humans_test = [r for r in pure_test if r.label is DocLabel.HUMAN]

    print("── held-out test split (seen families) ──")
    results["seen_overall"] = doc_metrics_line("all seen families", pure_test, system)

    per_family: list[dict[str, object]] = []
    families = sorted({r.model_family for r in pure_test if r.model_family is not None})
    for family in families:
        family_ai = [r for r in pure_test if r.model_family == family]
        subset = family_ai + humans_test
        per_family.append(doc_metrics_line(f"family: {family}", subset, system))
    results["seen_per_family"] = per_family

    mixed_test = [r for r in test if r.label is DocLabel.MIXED]
    if mixed_test:
        loc = evaluate_localization(mixed_test, system)
        mixed = evaluate_mixed_doc_labels(mixed_test, system)
        loc_auroc = f"{loc.auroc:.3f}" if loc.auroc is not None else "n/a"
        print(
            f"  {'localization (mixed docs)':28s} n={loc.n:5d}  f1={loc.f1:.3f}  auroc={loc_auroc}"
        )
        frac = f"{mixed.frac_labelled_mixed:.0%}"
        print(f"  {'mixed docs labelled MIXED':28s} n={mixed.n:5d}  frac={frac}")
        results["localization"] = loc.model_dump()
        results["mixed_doc_labels"] = mixed.model_dump()

    print("── GPT-4 OOD testbed (unseen family + unseen domain) ──")
    ood = read_jsonl(args.ood)
    results["ood_gpt4"] = doc_metrics_line("gpt-4 (unseen)", ood, system)

    print("── GPT-4 OOD + paraphrase attack ──")
    ood_para = read_jsonl(args.ood_para)
    results["ood_gpt4_paraphrased"] = doc_metrics_line("gpt-4 paraphrased", ood_para, system)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"metrics written to {args.out}")


if __name__ == "__main__":
    main()
