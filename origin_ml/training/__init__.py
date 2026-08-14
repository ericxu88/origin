"""Training entry points for Origin detectors."""

from origin_ml.training.classical import (
    extract_feature_matrix,
    train_baseline,
    train_sentence_baseline,
)
from origin_ml.training.examples import sentence_examples
from origin_ml.training.neural import NeuralTrainConfig, TrainReport, train_neural

__all__ = [
    "NeuralTrainConfig",
    "TrainReport",
    "extract_feature_matrix",
    "sentence_examples",
    "train_baseline",
    "train_neural",
    "train_sentence_baseline",
]
