"""Research evaluation: metrics, slices, ablations, experiments (SPEC §5)."""

from origin_ml.evaluation.ablations import ABLATION_NAMES, build_ablation_system
from origin_ml.evaluation.evaluate import (
    AblationSystem,
    MixedDocSummary,
    evaluate_doc_classification,
    evaluate_localization,
    evaluate_mixed_doc_labels,
    sentence_truth,
)
from origin_ml.evaluation.experiment import (
    AblationResult,
    ExperimentConfig,
    ExperimentResult,
    run_experiment,
)
from origin_ml.evaluation.metrics import (
    CalibrationBin,
    ClassificationMetrics,
    ConfusionMatrix,
    compute_binary_metrics,
)

__all__ = [
    "ABLATION_NAMES",
    "AblationResult",
    "AblationSystem",
    "CalibrationBin",
    "ClassificationMetrics",
    "ConfusionMatrix",
    "ExperimentConfig",
    "ExperimentResult",
    "MixedDocSummary",
    "build_ablation_system",
    "compute_binary_metrics",
    "evaluate_doc_classification",
    "evaluate_localization",
    "evaluate_mixed_doc_labels",
    "run_experiment",
    "sentence_truth",
]
