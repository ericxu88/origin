"""Training entry points for Origin detectors."""

from origin_ml.training.classical import extract_feature_matrix, train_baseline
from origin_ml.training.neural import (
    NeuralTrainConfig,
    TrainReport,
    sentence_examples,
    train_neural,
)

__all__ = [
    "NeuralTrainConfig",
    "TrainReport",
    "extract_feature_matrix",
    "sentence_examples",
    "train_baseline",
    "train_neural",
]
