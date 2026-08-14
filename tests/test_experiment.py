"""End-to-end experiment + ablation tests on the sample corpus (SPEC E-3..E-7)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from origin_ml.evaluation import (
    ABLATION_NAMES,
    ExperimentConfig,
    ExperimentResult,
    run_experiment,
)

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "experiments" / "sample.json"


@pytest.fixture(scope="module")
def result(tmp_path_factory: pytest.TempPathFactory) -> tuple[ExperimentResult, Path]:
    """Run a reduced-but-complete experiment (all 5 ablations, 1 holdout family)."""
    config = ExperimentConfig.from_file(CONFIG_PATH).model_copy(
        update={
            "dataset": str(ROOT / "data" / "sample" / "documents.jsonl"),
            "neural_checkpoint": str(ROOT / "tests" / "fixtures" / "tiny_classifier"),
            "output_dir": str(tmp_path_factory.mktemp("runs")),
            "neural_epochs": 2,
        }
    )
    return run_experiment(config), Path(config.output_dir) / config.name


class TestExperimentEndToEnd:
    def test_all_ablations_ran(self, result: tuple[ExperimentResult, Path]) -> None:
        res, _ = result
        assert set(res.ablations) == set(ABLATION_NAMES)

    def test_classical_full_performs_on_seen_docs(
        self, result: tuple[ExperimentResult, Path]
    ) -> None:
        res, _ = result
        seen = res.ablations["classical_full"].seen
        assert seen.auroc is not None and seen.auroc > 0.9
        assert seen.accuracy > 0.85

    def test_localization_metrics_present(self, result: tuple[ExperimentResult, Path]) -> None:
        res, _ = result
        loc = res.ablations["classical_full"].localization
        assert loc.n > 20  # sentences across mixed docs
        assert loc.auroc is not None and loc.auroc > 0.8

    def test_paraphrase_slice_evaluated(self, result: tuple[ExperimentResult, Path]) -> None:
        res, _ = result
        assert any(a.paraphrased is not None for a in res.ablations.values())

    def test_unseen_family_slice(self, result: tuple[ExperimentResult, Path]) -> None:
        res, _ = result
        unseen = res.ablations["classical_full"].unseen_family
        assert "gamma" in unseen
        assert unseen["gamma"].n > 0
        # Statistical fixture signals transfer across fictional families.
        assert unseen["gamma"].auroc is not None and unseen["gamma"].auroc > 0.8

    def test_mixed_doc_label_summary(self, result: tuple[ExperimentResult, Path]) -> None:
        res, _ = result
        summary = res.ablations["classical_full"].mixed_doc_labels
        assert summary.n > 0
        assert 0.0 <= summary.frac_labelled_mixed <= 1.0

    def test_rq6_correlations_present(self, result: tuple[ExperimentResult, Path]) -> None:
        res, _ = result
        assert "neural_vs_classical_prob" in res.rq6_correlations
        assert -1.0 <= res.rq6_correlations["neural_vs_classical_prob"] <= 1.0

    def test_machine_readable_output(self, result: tuple[ExperimentResult, Path]) -> None:
        _, out_dir = result
        payload = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
        assert payload["name"] == "sample-full"
        assert payload["git_commit"]
        assert payload["config"]["seed"] == 0
        assert set(payload["ablations"]) == set(ABLATION_NAMES)

    def test_plots_written(self, result: tuple[ExperimentResult, Path]) -> None:
        _, out_dir = result
        for name in ("ablation_auroc", "robustness_f1", "calibration_classical_full"):
            assert (out_dir / f"{name}.png").exists()
            assert (out_dir / f"{name}.svg").exists()

    def test_committed_config_is_valid(self) -> None:
        config = ExperimentConfig.from_file(CONFIG_PATH)
        assert config.name == "sample-full"
        assert set(config.ablations) == set(ABLATION_NAMES)


class TestConfigValidation:
    def test_unknown_scorer_rejected(self, result: tuple[ExperimentResult, Path]) -> None:
        res, _ = result
        bad = res.config.model_copy(update={"scorer": "bogus"})
        with pytest.raises(ValueError, match="unknown scorer"):
            run_experiment(bad, write_outputs=False)

    def test_unknown_ablation_rejected(self, result: tuple[ExperimentResult, Path]) -> None:
        res, _ = result
        bad = res.config.model_copy(update={"ablations": ("nonsense",)})
        with pytest.raises(ValueError, match="unknown ablation"):
            run_experiment(bad, write_outputs=False)
