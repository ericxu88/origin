"""CLI tests (SPEC CLI-1..CLI-5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from origin_ml.cli import app

runner = CliRunner()
ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "data" / "sample" / "documents.jsonl"
TINY_CLASSIFIER = ROOT / "tests" / "fixtures" / "tiny_classifier"

MIXED_TEXT = (
    "The crooked lantern flickered near the abandoned mill. "
    "Additionally, the system provides a clear and effective way to improve overall results. "
    "Rain again."
)


@pytest.fixture(scope="module")
def artifacts_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train baselines once via the real CLI command (CLI-3)."""
    out = tmp_path_factory.mktemp("artifacts")
    result = runner.invoke(
        app,
        ["train", "baseline", "--dataset", str(SAMPLE), "--out", str(out), "--seed", "0"],
    )
    assert result.exit_code == 0, result.output
    return out


class TestFeaturesCommand:
    def test_json_output_with_stub_scorer(self) -> None:
        result = runner.invoke(app, ["features", "--text", "Hello world. Goodbye moon.", "--json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["sent.count"] == 2.0
        assert "ppl.doc_perplexity" in payload

    def test_scorer_none_drops_lm_features(self) -> None:
        result = runner.invoke(
            app, ["features", "--text", "Hello world.", "--scorer", "none", "--json"]
        )
        assert result.exit_code == 0, result.output
        assert not any(key.startswith("ppl.") for key in json.loads(result.output))

    def test_reads_file(self, tmp_path: Path) -> None:
        doc = tmp_path / "doc.txt"
        doc.write_text("One sentence here.", encoding="utf-8")
        result = runner.invoke(app, ["features", str(doc), "--json"])
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["sent.count"] == 1.0

    def test_requires_exactly_one_input(self) -> None:
        assert runner.invoke(app, ["features"]).exit_code != 0
        assert runner.invoke(app, ["features", "somefile", "--text", "x"]).exit_code != 0

    def test_unknown_scorer_rejected(self) -> None:
        assert runner.invoke(app, ["features", "--text", "x", "--scorer", "bogus"]).exit_code != 0


class TestTrainBaselineCommand:
    def test_writes_artifacts(self, artifacts_dir: Path) -> None:
        assert (artifacts_dir / "doc_baseline.json").exists()
        assert (artifacts_dir / "sentence_baseline.json").exists()

    def test_artifacts_load_and_carry_metadata(self, artifacts_dir: Path) -> None:
        from origin_ml.detectors.classical import BaselineDetector

        doc_model = BaselineDetector.load(artifacts_dir / "doc_baseline.json")
        sent_model = BaselineDetector.load(artifacts_dir / "sentence_baseline.json")
        assert doc_model.training_meta["level"] == "document"
        assert sent_model.training_meta["level"] == "sentence"
        assert doc_model.is_calibrated


class TestAnalyzeCommand:
    def test_human_readable_output(self, artifacts_dir: Path) -> None:
        result = runner.invoke(
            app,
            ["analyze", "--text", MIXED_TEXT, "--artifacts", str(artifacts_dir)],
        )
        assert result.exit_code == 0, result.output
        assert "verdict" in result.output
        assert "P(ai)" in result.output
        assert "not proof" in result.output  # disclaimer (G-1)
        assert result.output.count("[") >= 3  # per-sentence probability rows

    def test_json_output(self, artifacts_dir: Path) -> None:
        result = runner.invoke(
            app,
            ["analyze", "--text", MIXED_TEXT, "--artifacts", str(artifacts_dir), "--json"],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["label"] in ("human", "ai", "mixed")
        assert len(payload["evidence"]["heatmap"]) == 3
        assert payload["evidence"]["heatmap"][0]["kind"] == "model"

    def test_demo_training_fallback(self) -> None:
        result = runner.invoke(app, ["analyze", "--text", MIXED_TEXT, "--dataset", str(SAMPLE)])
        assert result.exit_code == 0, result.output
        assert "verdict" in result.output

    def test_reads_file(self, artifacts_dir: Path, tmp_path: Path) -> None:
        doc = tmp_path / "doc.txt"
        doc.write_text(MIXED_TEXT, encoding="utf-8")
        result = runner.invoke(app, ["analyze", str(doc), "--artifacts", str(artifacts_dir)])
        assert result.exit_code == 0, result.output

    def test_empty_document_fails_cleanly(self, artifacts_dir: Path) -> None:
        result = runner.invoke(app, ["analyze", "--text", "   ", "--artifacts", str(artifacts_dir)])
        assert result.exit_code == 1


class TestEvaluateCommand:
    def test_evaluates_split_and_writes_json(self, artifacts_dir: Path, tmp_path: Path) -> None:
        out = tmp_path / "metrics.json"
        result = runner.invoke(
            app,
            [
                "evaluate",
                "--artifacts",
                str(artifacts_dir),
                "--dataset",
                str(SAMPLE),
                "--split",
                "test",
                "--out",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "documents" in result.output
        assert "auroc" in result.output
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["documents"]["n"] > 0
        assert payload["documents"]["auroc"] is not None
        assert payload["localization"]["n"] > 0

    def test_bad_split_fails(self, artifacts_dir: Path) -> None:
        result = runner.invoke(
            app,
            ["evaluate", "--artifacts", str(artifacts_dir), "--split", "val"],
        )
        assert result.exit_code == 1


class TestTrainNeuralCommand:
    def test_trains_and_saves_checkpoint(self, tmp_path: Path) -> None:
        out = tmp_path / "neural"
        result = runner.invoke(
            app,
            [
                "train",
                "neural",
                "--checkpoint",
                str(TINY_CLASSIFIER),
                "--dataset",
                str(SAMPLE),
                "--out",
                str(out),
                "--epochs",
                "1",
                "--device",
                "cpu",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "epoch losses" in result.output
        assert (out / "config.json").exists()


class TestExperimentCommand:
    def test_runs_reduced_experiment(self, tmp_path: Path) -> None:
        config = {
            "name": "cli-mini",
            "dataset": str(SAMPLE),
            "output_dir": str(tmp_path / "runs"),
            "seed": 0,
            "scorer": "stub",
            "neural_checkpoint": str(TINY_CLASSIFIER),
            "neural_epochs": 1,
            "ablations": ["classical_full"],
            "holdout_families": [],
        }
        config_path = tmp_path / "mini.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        result = runner.invoke(app, ["experiment", str(config_path)])
        assert result.exit_code == 0, result.output
        assert "classical_full" in result.output
        assert (tmp_path / "runs" / "cli-mini" / "results.json").exists()


class TestVersionCommand:
    def test_version(self) -> None:
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert result.output.strip()
