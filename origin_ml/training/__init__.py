"""Training entry points for Origin detectors."""

from origin_ml.training.classical import extract_feature_matrix, train_baseline

__all__ = ["extract_feature_matrix", "train_baseline"]
