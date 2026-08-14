"""FeaturePipeline composition tests (SPEC F-12)."""

from __future__ import annotations

import pytest

from origin_ml.features import (
    FeaturePipeline,
    FeatureVector,
    PerplexityExtractor,
    StyleExtractor,
    build_default_pipeline,
)
from origin_ml.scoring import FixedScorer


class TestDefaultPipeline:
    def test_with_scorer_includes_lm_features(self) -> None:
        pipeline = build_default_pipeline(scorer=FixedScorer({}))
        assert any(n.startswith("ppl.") for n in pipeline.feature_names)
        vector = pipeline.extract("Hello world. Goodbye moon.")
        assert vector.names == pipeline.feature_names
        assert len(vector.values) == len(vector.names)

    def test_without_scorer_excludes_lm_features(self) -> None:
        pipeline = build_default_pipeline(scorer=None)
        assert not any(n.startswith("ppl.") for n in pipeline.feature_names)
        vector = pipeline.extract("Hello world.")
        assert "sent.count" in vector.names

    def test_names_are_unique_and_stable(self) -> None:
        pipeline = build_default_pipeline(scorer=FixedScorer({}))
        assert len(set(pipeline.feature_names)) == len(pipeline.feature_names)
        again = build_default_pipeline(scorer=FixedScorer({}))
        assert again.feature_names == pipeline.feature_names


class TestValidation:
    def test_scoring_extractor_requires_scorer_at_construction(self) -> None:
        with pytest.raises(ValueError, match="require a scorer"):
            FeaturePipeline([PerplexityExtractor()], scorer=None)

    def test_duplicate_feature_names_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate feature names"):
            FeaturePipeline([StyleExtractor(), StyleExtractor()])

    def test_empty_pipeline_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one extractor"):
            FeaturePipeline([])


class TestFeatureVector:
    def test_as_dict_and_getitem(self) -> None:
        vector = FeatureVector(names=("a", "b"), values=(1.0, 2.0))
        assert vector.as_dict() == {"a": 1.0, "b": 2.0}
        assert vector["b"] == 2.0
        with pytest.raises(KeyError):
            vector["missing"]

    def test_length_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            FeatureVector(names=("a",), values=(1.0, 2.0))
