"""Experiment runner: ablations x evaluation slices → JSON + plots (SPEC E-3..E-7).

An experiment is described by a JSON config (see ``experiments/sample.json``),
runs fully offline on the sample corpus, and writes:

- ``results.json`` — machine-readable metrics per ablation and slice, with the
  config, seed, and git commit embedded (SPEC E-5),
- publication-quality matplotlib figures (SPEC E-5),

into ``<output_dir>/<experiment name>/``.

Slices per ablation (SPEC E-3):

- ``seen``            — pure human/ai test documents (in-distribution),
- ``paraphrased``     — paraphrase-transformed test documents,
- ``localization``    — sentence-level metrics over mixed test documents,
- ``mixed_doc_labels``— how often mixed docs get the MIXED label,
- ``unseen_family:<f>`` — retrained without family ``f``, evaluated on it
  (SPEC DS-4 / RQ4).

RQ6 (SPEC E-7): Pearson correlations between the neural system's sentence
probabilities and classical per-sentence features/probabilities.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from origin_ml.datasets.io import read_jsonl
from origin_ml.datasets.schema import DocLabel, DocumentRecord
from origin_ml.datasets.splits import assign_splits, family_holdout_split
from origin_ml.evaluation.ablations import ABLATION_NAMES, build_ablation_system
from origin_ml.evaluation.evaluate import (
    AblationSystem,
    MixedDocSummary,
    evaluate_doc_classification,
    evaluate_localization,
    evaluate_mixed_doc_labels,
)
from origin_ml.evaluation.metrics import ClassificationMetrics
from origin_ml.features.pipeline import build_default_pipeline
from origin_ml.scoring.base import Scorer
from origin_ml.scoring.stub import StubScorer

__all__ = ["AblationResult", "ExperimentConfig", "ExperimentResult", "run_experiment"]


class ExperimentConfig(BaseModel):
    """Declarative experiment description (SPEC E-4, R-2)."""

    model_config = ConfigDict(frozen=True)

    name: str
    dataset: str
    output_dir: str = "runs"
    seed: int = 0
    train_ratio: float = 0.8
    scorer: str = "stub"  # "stub", "none", or "hf:<checkpoint>"
    neural_checkpoint: str
    neural_epochs: int = 3
    ablations: tuple[str, ...] = ABLATION_NAMES
    holdout_families: tuple[str, ...] = ()

    @classmethod
    def from_file(cls, path: Path) -> ExperimentConfig:
        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))


class AblationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    description: str
    seen: ClassificationMetrics
    paraphrased: ClassificationMetrics | None
    localization: ClassificationMetrics
    mixed_doc_labels: MixedDocSummary
    unseen_family: dict[str, ClassificationMetrics]


class ExperimentResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    created_at: str
    git_commit: str
    seed: int
    config: ExperimentConfig
    n_train: int
    n_test: int
    ablations: dict[str, AblationResult]
    rq6_correlations: dict[str, float] = Field(default_factory=dict)


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _make_scorer(spec: str) -> Scorer | None:
    if spec == "stub":
        return StubScorer()
    if spec == "none":
        return None
    if spec.startswith("hf:"):
        from origin_ml.scoring.hf import HFCausalScorer

        return HFCausalScorer(spec.removeprefix("hf:"))
    raise ValueError(f"unknown scorer spec {spec!r} (expected 'stub', 'none', or 'hf:<ckpt>')")


def _evaluate_system(
    system: AblationSystem,
    test_records: list[DocumentRecord],
) -> tuple[
    ClassificationMetrics, ClassificationMetrics | None, ClassificationMetrics, MixedDocSummary
]:
    pure = [r for r in test_records if r.label in (DocLabel.HUMAN, DocLabel.AI)]
    seen = evaluate_doc_classification([r for r in pure if not r.is_paraphrase] or pure, system)
    paraphrased_records = [r for r in pure if r.is_paraphrase]
    paraphrased = (
        evaluate_doc_classification(paraphrased_records, system) if paraphrased_records else None
    )
    localization = evaluate_localization(test_records, system)
    mixed_labels = evaluate_mixed_doc_labels(test_records, system)
    return seen, paraphrased, localization, mixed_labels


def _rq6_correlations(
    neural: AblationSystem,
    classical: AblationSystem,
    test_records: list[DocumentRecord],
    scorer: Scorer | None,
) -> dict[str, float]:
    """Pearson correlation of neural sentence P(ai) with classical signals (RQ6)."""
    pipeline = build_default_pipeline(scorer=scorer)
    neural_p: list[float] = []
    classical_p: list[float] = []
    feature_rows: list[tuple[float, ...]] = []
    for record in test_records[:40]:
        n_scores = neural.sentence_p_ai(record.text)
        c_scores = classical.sentence_p_ai(record.text)
        for (span, p_n), (_, p_c) in zip(n_scores, c_scores, strict=True):
            neural_p.append(p_n)
            classical_p.append(p_c)
            feature_rows.append(pipeline.extract(span.text).values)

    correlations: dict[str, float] = {}

    def corr(xs: list[float], ys: list[float]) -> float:
        if len(set(xs)) < 2 or len(set(ys)) < 2:
            return 0.0
        return float(np.corrcoef(xs, ys)[0, 1])

    correlations["neural_vs_classical_prob"] = corr(neural_p, classical_p)
    features = np.asarray(feature_rows, dtype=np.float64)
    for i, name in enumerate(pipeline.feature_names):
        if name in (
            "ppl.doc_perplexity",
            "ppl.mean_surprisal_bits",
            "lex.ttr",
            "sent.mean_len_words",
        ):
            correlations[f"neural_vs_{name}"] = corr(neural_p, list(features[:, i]))
    return correlations


def run_experiment(config: ExperimentConfig, *, write_outputs: bool = True) -> ExperimentResult:
    """Run all configured ablations and slices (SPEC E-6: single entry point)."""
    records = read_jsonl(Path(config.dataset))
    split_records = assign_splits(
        records, train=config.train_ratio, val=0.0, test=1.0 - config.train_ratio, seed=config.seed
    )
    train_records = [r for r in split_records if r.split == "train"]
    test_records = [r for r in split_records if r.split == "test"]
    scorer = _make_scorer(config.scorer)

    systems: dict[str, AblationSystem] = {}
    results: dict[str, AblationResult] = {}
    for name in config.ablations:
        system = build_ablation_system(
            name,
            train_records,
            scorer=scorer,
            neural_checkpoint=config.neural_checkpoint,
            seed=config.seed,
            neural_epochs=config.neural_epochs,
        )
        systems[name] = system
        seen, paraphrased, localization, mixed_labels = _evaluate_system(system, test_records)

        unseen: dict[str, ClassificationMetrics] = {}
        for family in config.holdout_families:
            holdout_train, holdout_test = family_holdout_split(records, {family}, seed=config.seed)
            holdout_system = build_ablation_system(
                name,
                holdout_train,
                scorer=scorer,
                neural_checkpoint=config.neural_checkpoint,
                seed=config.seed,
                neural_epochs=config.neural_epochs,
            )
            unseen[family] = evaluate_doc_classification(
                [r for r in holdout_test if r.label in (DocLabel.HUMAN, DocLabel.AI)],
                holdout_system,
            )

        results[name] = AblationResult(
            description=system.description,
            seen=seen,
            paraphrased=paraphrased,
            localization=localization,
            mixed_doc_labels=mixed_labels,
            unseen_family=unseen,
        )

    rq6: dict[str, float] = {}
    if "neural" in systems and "classical_full" in systems:
        rq6 = _rq6_correlations(systems["neural"], systems["classical_full"], test_records, scorer)

    result = ExperimentResult(
        name=config.name,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        git_commit=_git_commit(),
        seed=config.seed,
        config=config,
        n_train=len(train_records),
        n_test=len(test_records),
        ablations=results,
        rq6_correlations=rq6,
    )

    if write_outputs:
        out_dir = Path(config.output_dir) / config.name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "results.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
        from origin_ml.evaluation.plots import write_experiment_plots

        write_experiment_plots(result, out_dir)
    return result
