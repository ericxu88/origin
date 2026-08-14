"""Origin command-line interface (SPEC §8).

Commands:

- ``origin analyze``        — detect + localize + explain a document (CLI-1)
- ``origin features``       — extract the statistical feature vector (CLI-2)
- ``origin train baseline`` — train classical document+sentence models (CLI-3)
- ``origin train neural``   — fine-tune the neural sentence classifier (CLI-5)
- ``origin evaluate``       — evaluate trained artifacts on a dataset split (CLI-4)
- ``origin experiment``     — run a full ablation experiment config (E-6)
- ``origin version``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import typer

import origin_ml
from origin_ml.features.pipeline import FeaturePipeline, build_default_pipeline
from origin_ml.scoring.base import Scorer
from origin_ml.scoring.stub import StubScorer

app = typer.Typer(
    name="origin",
    help="Origin: interpretable detection and explanation of LLM-generated text.",
    no_args_is_help=True,
    add_completion=False,
)
train_app = typer.Typer(help="Train Origin detectors.", no_args_is_help=True)
app.add_typer(train_app, name="train")

DEFAULT_DATASET = Path("data/sample/documents.jsonl")

ScorerOption = Annotated[
    str,
    typer.Option(
        "--scorer",
        help="LM scorer: 'stub' (deterministic, offline), 'none', or 'hf:<checkpoint>'.",
    ),
]


def _make_scorer(spec: str) -> Scorer | None:
    if spec == "stub":
        return StubScorer()
    if spec == "none":
        return None
    if spec.startswith("hf:"):
        from origin_ml.scoring.hf import HFCausalScorer

        return HFCausalScorer(spec.removeprefix("hf:"))
    raise typer.BadParameter(f"unknown scorer '{spec}' (expected 'stub', 'none', or 'hf:<ckpt>')")


def _read_text(path: Path | None, text: str | None) -> str:
    if (path is None) == (text is None):
        raise typer.BadParameter("provide exactly one of PATH or --text")
    if path is not None:
        if str(path) == "-":
            return sys.stdin.read()
        if not path.exists():
            raise typer.BadParameter(f"file not found: {path}")
        return path.read_text(encoding="utf-8")
    assert text is not None
    return text


def _pipeline(scorer_spec: str) -> FeaturePipeline:
    return build_default_pipeline(scorer=_make_scorer(scorer_spec))


def _load_or_train_baselines(
    artifacts: Path | None, dataset: Path, pipeline: FeaturePipeline, seed: int
) -> tuple[object, object]:
    """Return (doc_baseline, sentence_baseline), loading artifacts or demo-training."""
    from origin_ml.detectors.classical import BaselineDetector

    if artifacts is not None:
        return (
            BaselineDetector.load(artifacts / "doc_baseline.json"),
            BaselineDetector.load(artifacts / "sentence_baseline.json"),
        )
    from origin_ml.datasets import assign_splits, read_jsonl
    from origin_ml.training import train_baseline, train_sentence_baseline

    typer.echo(f"[origin] no --artifacts given; training demo baselines on {dataset}", err=True)
    records = assign_splits(read_jsonl(dataset), seed=seed)
    return (
        train_baseline(records, pipeline, dataset_name=str(dataset)),
        train_sentence_baseline(records, pipeline, dataset_name=str(dataset)),
    )


@app.command()
def version() -> None:
    """Print the Origin version."""
    typer.echo(origin_ml.__version__)


@app.command()
def features(
    path: Annotated[
        Path | None,
        typer.Argument(help="Text file to analyze, or '-' for stdin."),
    ] = None,
    text: Annotated[str | None, typer.Option("--text", help="Analyze a literal string.")] = None,
    scorer: ScorerOption = "stub",
    json_out: Annotated[bool, typer.Option("--json", help="Emit JSON instead of a table.")] = False,
) -> None:
    """Extract Origin's statistical feature vector from a document (CLI-2)."""
    document = _read_text(path, text)
    vector = _pipeline(scorer).extract(document)
    if json_out:
        typer.echo(json.dumps(vector.as_dict(), indent=2))
    else:
        width = max(len(name) for name in vector.names)
        for name, value in zip(vector.names, vector.values, strict=True):
            typer.echo(f"{name:<{width}}  {value: .6f}")


@app.command()
def analyze(
    path: Annotated[
        Path | None, typer.Argument(help="Text file to analyze, or '-' for stdin.")
    ] = None,
    text: Annotated[str | None, typer.Option("--text", help="Analyze a literal string.")] = None,
    scorer: ScorerOption = "stub",
    artifacts: Annotated[
        Path | None,
        typer.Option(help="Directory with doc_baseline.json + sentence_baseline.json."),
    ] = None,
    neural_checkpoint: Annotated[
        str | None, typer.Option("--neural", help="Neural checkpoint for sentence probabilities.")
    ] = None,
    dataset: Annotated[
        Path, typer.Option(help="Corpus for demo training when no artifacts are given.")
    ] = DEFAULT_DATASET,
    seed: Annotated[int, typer.Option(help="Seed for demo training splits.")] = 0,
    json_out: Annotated[
        bool, typer.Option("--json", help="Emit the full AnalysisResult as JSON.")
    ] = False,
) -> None:
    """Detect, localize, and explain AI-generated text in a document (CLI-1)."""
    from origin_ml.detectors.classical import BaselineDetector
    from origin_ml.explainability import analyze_document

    document = _read_text(path, text)
    pipeline = _pipeline(scorer)

    neural = None
    if neural_checkpoint is not None:
        from origin_ml.detectors.neural import NeuralDetector

        neural = NeuralDetector.from_checkpoint(neural_checkpoint)

    doc_model, sent_model = _load_or_train_baselines(artifacts, dataset, pipeline, seed)
    assert isinstance(doc_model, BaselineDetector)
    assert isinstance(sent_model, BaselineDetector)
    try:
        result = analyze_document(
            document,
            pipeline=pipeline,
            sentence_baseline=sent_model,
            doc_baseline=doc_model,
            neural=neural,
        )
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if json_out:
        typer.echo(result.model_dump_json(indent=2))
        return

    typer.echo(f"verdict     : {result.label.value.upper()}  (confidence {result.confidence:.0%})")
    scores = result.class_probabilities
    typer.echo(
        f"class scores: human {scores.human:.0%} / ai {scores.ai:.0%} / mixed {scores.mixed:.0%}"
    )
    typer.echo(
        f"sentence AI : mean P(ai) {result.mean_p_ai:.0%}, "
        f"{result.frac_ai_sentences:.0%} of sentences AI-leaning"
    )
    if result.document_p_ai is not None:
        typer.echo(f"document AI : P(ai) {result.document_p_ai:.0%}")
    typer.echo(f"detector    : {result.detector}")
    typer.echo("")
    for heat in result.evidence.heatmap:
        bar = "#" * round(heat.p_ai * 20)
        typer.echo(f"  [{heat.p_ai:>5.0%}] {bar:<20} {heat.text}")
    typer.echo("")
    typer.echo(f"note: {result.disclaimer}")


@train_app.command("baseline")
def train_baseline_cmd(
    dataset: Annotated[Path, typer.Option(help="JSONL dataset to train on.")] = DEFAULT_DATASET,
    out: Annotated[Path, typer.Option(help="Output directory for artifacts.")] = Path(
        "artifacts/baseline"
    ),
    scorer: ScorerOption = "stub",
    seed: Annotated[int, typer.Option(help="Split + training seed.")] = 0,
    train_ratio: Annotated[float, typer.Option(help="Train split fraction.")] = 0.8,
) -> None:
    """Train the classical document + sentence baselines and save artifacts (CLI-3)."""
    from origin_ml.datasets import assign_splits, read_jsonl
    from origin_ml.detectors.classical import BaselineConfig
    from origin_ml.training import train_baseline, train_sentence_baseline

    pipeline = _pipeline(scorer)
    records = assign_splits(
        read_jsonl(dataset), train=train_ratio, val=0.0, test=1.0 - train_ratio, seed=seed
    )
    config = BaselineConfig(seed=seed)
    doc_model = train_baseline(records, pipeline, config=config, dataset_name=str(dataset))
    sent_model = train_sentence_baseline(
        records, pipeline, config=config, dataset_name=str(dataset)
    )
    doc_model.save(out / "doc_baseline.json")
    sent_model.save(out / "sentence_baseline.json")
    typer.echo(f"trained document baseline  : {doc_model.training_meta}")
    typer.echo(f"trained sentence baseline  : {sent_model.training_meta}")
    typer.echo(f"artifacts written to {out}/")
    top = doc_model.feature_importances()[:5]
    typer.echo("top document features (|coef|):")
    for importance in top:
        typer.echo(f"  {importance.coefficient:+.3f}  {importance.name}")


@train_app.command("neural")
def train_neural_cmd(
    checkpoint: Annotated[str, typer.Option(help="Base HF checkpoint (path or hub id).")],
    dataset: Annotated[Path, typer.Option(help="JSONL dataset to train on.")] = DEFAULT_DATASET,
    out: Annotated[Path, typer.Option(help="Output checkpoint directory.")] = Path(
        "artifacts/neural"
    ),
    epochs: Annotated[int, typer.Option(help="Training epochs.")] = 3,
    lr: Annotated[float, typer.Option(help="Learning rate.")] = 5e-4,
    batch_size: Annotated[int, typer.Option(help="Batch size.")] = 8,
    seed: Annotated[int, typer.Option(help="Seed.")] = 0,
    device: Annotated[
        str | None, typer.Option(help="Torch device (default: auto CUDA/MPS/CPU).")
    ] = None,
    train_ratio: Annotated[float, typer.Option(help="Train split fraction.")] = 0.8,
) -> None:
    """Fine-tune the sentence-granular neural detector (CLI-5)."""
    from origin_ml.datasets import assign_splits, read_jsonl
    from origin_ml.training import NeuralTrainConfig, train_neural

    records = assign_splits(
        read_jsonl(dataset), train=train_ratio, val=0.0, test=1.0 - train_ratio, seed=seed
    )
    train_records = [r for r in records if r.split == "train"]
    config = NeuralTrainConfig(
        checkpoint=checkpoint,
        output_dir=out,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        seed=seed,
        device=device,
    )
    _, report = train_neural(train_records, config)
    losses = ", ".join(f"{loss:.4f}" for loss in report.epoch_losses)
    typer.echo(f"trained on {report.n_examples} sentences ({report.device})")
    typer.echo(f"epoch losses: {losses}")
    typer.echo(f"checkpoint written to {report.saved_to}")


@app.command()
def evaluate(
    artifacts: Annotated[
        Path, typer.Option(help="Directory with doc_baseline.json + sentence_baseline.json.")
    ],
    dataset: Annotated[Path, typer.Option(help="JSONL dataset to evaluate on.")] = DEFAULT_DATASET,
    split: Annotated[
        str, typer.Option(help="Split to evaluate ('test', 'train', or 'all').")
    ] = "test",
    scorer: ScorerOption = "stub",
    seed: Annotated[int, typer.Option(help="Split seed (must match training).")] = 0,
    train_ratio: Annotated[float, typer.Option(help="Train split fraction.")] = 0.8,
    out: Annotated[
        Path | None, typer.Option(help="Write machine-readable metrics JSON here.")
    ] = None,
) -> None:
    """Evaluate trained classical artifacts on a dataset split (CLI-4)."""
    import numpy as np

    from origin_ml.datasets import assign_splits, read_jsonl
    from origin_ml.detectors.classical import BaselineDetector
    from origin_ml.evaluation import (
        AblationSystem,
        evaluate_doc_classification,
        evaluate_localization,
        evaluate_mixed_doc_labels,
    )
    from origin_ml.evaluation.evaluate import default_sentence_scorer

    pipeline = _pipeline(scorer)
    doc_model = BaselineDetector.load(artifacts / "doc_baseline.json")
    sent_model = BaselineDetector.load(artifacts / "sentence_baseline.json")

    records = assign_splits(
        read_jsonl(dataset), train=train_ratio, val=0.0, test=1.0 - train_ratio, seed=seed
    )
    subset = records if split == "all" else [r for r in records if r.split == split]
    if not subset:
        typer.echo(f"error: split '{split}' selected no records", err=True)
        raise typer.Exit(code=1)

    def doc_p(text: str) -> float:
        return doc_model.predict_proba_one(
            np.asarray(pipeline.extract(text).values, dtype=np.float64)
        )

    def sentence_probs(sentences: list[str]) -> list[float]:
        matrix = np.asarray([pipeline.extract(s).values for s in sentences], dtype=np.float64)
        return [float(p) for p in sent_model.predict_proba(matrix)]

    system = AblationSystem(
        name="classical-artifacts",
        description=f"artifacts from {artifacts}",
        doc_p_ai=doc_p,
        sentence_p_ai=default_sentence_scorer(sentence_probs),
    )
    doc_metrics = evaluate_doc_classification(subset, system)
    payload: dict[str, object] = {
        "dataset": str(dataset),
        "split": split,
        "artifacts": str(artifacts),
        "documents": doc_metrics.model_dump(),
    }
    auroc = f"{doc_metrics.auroc:.3f}" if doc_metrics.auroc is not None else "n/a"
    typer.echo(
        f"documents  (n={doc_metrics.n}): acc={doc_metrics.accuracy:.3f} "
        f"f1={doc_metrics.f1:.3f} auroc={auroc} brier={doc_metrics.brier:.3f}"
    )

    if any(r.label.value == "mixed" for r in subset):
        loc = evaluate_localization(subset, system)
        mixed = evaluate_mixed_doc_labels(subset, system)
        loc_auroc = f"{loc.auroc:.3f}" if loc.auroc is not None else "n/a"
        typer.echo(
            f"sentences  (n={loc.n}): f1={loc.f1:.3f} auroc={loc_auroc} "
            f"(localization over mixed docs)"
        )
        typer.echo(f"mixed docs (n={mixed.n}): {mixed.frac_labelled_mixed:.0%} labelled MIXED")
        payload["localization"] = loc.model_dump()
        payload["mixed_doc_labels"] = mixed.model_dump()

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        typer.echo(f"metrics written to {out}")


@app.command()
def experiment(
    config_path: Annotated[Path, typer.Argument(help="Experiment JSON config.")],
) -> None:
    """Run a full ablation experiment from a config file (SPEC E-6)."""
    from origin_ml.evaluation import ExperimentConfig, run_experiment

    config = ExperimentConfig.from_file(config_path)
    result = run_experiment(config)
    typer.echo(f"experiment '{result.name}': {result.n_train} train / {result.n_test} test docs")
    for name, ablation in result.ablations.items():
        auroc = f"{ablation.seen.auroc:.3f}" if ablation.seen.auroc is not None else "n/a"
        typer.echo(f"  {name:22s} seen AUROC={auroc}")
    typer.echo(f"outputs: {Path(config.output_dir) / config.name}/")


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    main()
