"""Explainability + localization tests (SPEC L-2, L-3, X-1..X-4, G-1, G-2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from origin_ml.datasets import DocLabel, DocumentRecord, SegmentLabel, assign_splits, read_jsonl
from origin_ml.detectors import BaselineDetector, NeuralDetector
from origin_ml.explainability import DISCLAIMER, AnalysisResult, analyze_document
from origin_ml.features import FeaturePipeline, build_default_pipeline
from origin_ml.scoring import StubScorer
from origin_ml.training import train_baseline, train_sentence_baseline

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample" / "documents.jsonl"
TINY_CLASSIFIER = Path(__file__).resolve().parent / "fixtures" / "tiny_classifier"


@pytest.fixture(scope="module")
def pipeline() -> FeaturePipeline:
    return build_default_pipeline(scorer=StubScorer())


@pytest.fixture(scope="module")
def records() -> list[DocumentRecord]:
    return assign_splits(read_jsonl(SAMPLE), seed=0)


@pytest.fixture(scope="module")
def doc_baseline(records: list[DocumentRecord], pipeline: FeaturePipeline) -> BaselineDetector:
    return train_baseline(records, pipeline, dataset_name="sample")


@pytest.fixture(scope="module")
def sentence_baseline(records: list[DocumentRecord], pipeline: FeaturePipeline) -> BaselineDetector:
    return train_sentence_baseline(records, pipeline, dataset_name="sample")


@pytest.fixture(scope="module")
def result(
    pipeline: FeaturePipeline,
    sentence_baseline: BaselineDetector,
    doc_baseline: BaselineDetector,
) -> AnalysisResult:
    text = (
        "The crooked lantern flickered near the abandoned mill. "
        "Additionally, the system provides a clear and effective way to improve results. "
        "Rain again."
    )
    return analyze_document(
        text, pipeline=pipeline, sentence_baseline=sentence_baseline, doc_baseline=doc_baseline
    )


class TestClassicalPath:
    def test_human_documents_classified(
        self,
        records: list[DocumentRecord],
        pipeline: FeaturePipeline,
        sentence_baseline: BaselineDetector,
        doc_baseline: BaselineDetector,
    ) -> None:
        docs = [r for r in records if r.split == "test" and r.label is DocLabel.HUMAN][:5]
        assert docs
        for doc in docs:
            res = analyze_document(
                doc.text,
                pipeline=pipeline,
                sentence_baseline=sentence_baseline,
                doc_baseline=doc_baseline,
            )
            assert res.label is DocLabel.HUMAN, doc.id
            assert res.document_p_ai is not None and res.document_p_ai < 0.5

    def test_ai_documents_classified(
        self,
        records: list[DocumentRecord],
        pipeline: FeaturePipeline,
        sentence_baseline: BaselineDetector,
        doc_baseline: BaselineDetector,
    ) -> None:
        docs = [r for r in records if r.split == "test" and r.label is DocLabel.AI][:5]
        assert docs
        for doc in docs:
            res = analyze_document(
                doc.text,
                pipeline=pipeline,
                sentence_baseline=sentence_baseline,
                doc_baseline=doc_baseline,
            )
            assert res.label is DocLabel.AI, doc.id
            assert res.document_p_ai is not None and res.document_p_ai > 0.5

    def test_mixed_document_localization(
        self,
        records: list[DocumentRecord],
        pipeline: FeaturePipeline,
        sentence_baseline: BaselineDetector,
    ) -> None:
        """AI-span sentences must score above human-span sentences (SPEC L-2)."""
        mixed_docs = [r for r in records if r.label is DocLabel.MIXED][:5]
        assert mixed_docs
        for doc in mixed_docs:
            res = analyze_document(doc.text, pipeline=pipeline, sentence_baseline=sentence_baseline)

            def span_label(start: int, end: int, record: DocumentRecord) -> SegmentLabel | None:
                mid = (start + end) / 2
                for span in record.spans:
                    if span.start <= mid < span.end:
                        return span.label
                return None

            ai_probs = [
                h.p_ai
                for h in res.evidence.heatmap
                if span_label(h.start, h.end, doc) is SegmentLabel.AI
            ]
            human_probs = [
                h.p_ai
                for h in res.evidence.heatmap
                if span_label(h.start, h.end, doc) is SegmentLabel.HUMAN
            ]
            assert ai_probs and human_probs, doc.id
            assert float(np.mean(ai_probs)) > float(np.mean(human_probs)), doc.id

    def test_mixed_documents_labelled_mixed(
        self,
        records: list[DocumentRecord],
        pipeline: FeaturePipeline,
        sentence_baseline: BaselineDetector,
    ) -> None:
        mixed_docs = [r for r in records if r.label is DocLabel.MIXED]
        predictions = [
            analyze_document(d.text, pipeline=pipeline, sentence_baseline=sentence_baseline).label
            for d in mixed_docs
        ]
        # The aggregate rule needs a reasonable share of the fixture's mixed
        # docs recognized as mixed (block sizes vary, so a perfect score is
        # not expected).
        assert predictions.count(DocLabel.MIXED) >= len(mixed_docs) * 0.6


class TestEvidenceBundle:
    def test_heatmap_matches_sentences(self, result: AnalysisResult) -> None:
        assert len(result.evidence.heatmap) == 3
        for heat in result.evidence.heatmap:
            assert heat.end > heat.start
            assert 0.0 <= heat.p_ai <= 1.0

    def test_evidence_kind_tags(self, result: AnalysisResult) -> None:
        """SPEC X-4 / G-2: heuristic vs model tagging on every section."""
        assert all(h.kind == "model" for h in result.evidence.heatmap)
        assert all(s.kind == "heuristic" for s in result.evidence.sentence_statistics)
        assert result.evidence.document_features.kind == "heuristic"
        assert result.evidence.token_surprisals is not None
        assert result.evidence.token_surprisals.kind == "heuristic"
        assert result.evidence.distribution_comparison is not None
        assert result.evidence.distribution_comparison.kind == "heuristic"

    def test_token_surprisals_present_with_scorer(self, result: AnalysisResult) -> None:
        series = result.evidence.token_surprisals
        assert series is not None
        assert series.tokens
        assert all(t.surprisal_bits > 0 for t in series.tokens)

    def test_sentence_statistics(self, result: AnalysisResult) -> None:
        stats = result.evidence.sentence_statistics
        assert len(stats) == len(result.evidence.heatmap)
        for s in stats:
            assert s.n_words > 0
            assert s.perplexity is not None and s.perplexity > 1.0

    def test_distribution_comparison_populated(self, result: AnalysisResult) -> None:
        comparison = result.evidence.distribution_comparison
        assert comparison is not None
        features = {c.feature for c in comparison.comparisons}
        assert "ppl.doc_perplexity" in features
        assert all(c.closer_to in ("human", "ai", "similar") for c in comparison.comparisons)

    def test_disclaimer_always_present(self, result: AnalysisResult) -> None:
        assert result.disclaimer == DISCLAIMER
        assert "not proof" in result.disclaimer

    def test_json_serializable(self, result: AnalysisResult) -> None:
        assert '"heatmap"' in result.model_dump_json()


class TestVariants:
    def test_no_scorer_pipeline_degrades_gracefully(self, records: list[DocumentRecord]) -> None:
        plain = build_default_pipeline(scorer=None)
        sentence_model = train_sentence_baseline(records, plain, dataset_name="sample-plain")
        res = analyze_document(
            "One short sentence. Another one follows here.",
            pipeline=plain,
            sentence_baseline=sentence_model,
        )
        assert res.evidence.token_surprisals is None
        assert all(s.perplexity is None for s in res.evidence.sentence_statistics)
        assert res.evidence.document_features.scorer is None

    def test_neural_path(self, pipeline: FeaturePipeline) -> None:
        neural = NeuralDetector.from_checkpoint(TINY_CLASSIFIER, device="cpu")
        res = analyze_document(
            "One sentence here. Another sentence there.", pipeline=pipeline, neural=neural
        )
        assert res.detector.startswith("neural(")
        assert len(res.evidence.heatmap) == 2
        assert res.document_p_ai is None  # no doc baseline given

    def test_requires_sentence_capable_detector(
        self, pipeline: FeaturePipeline, doc_baseline: BaselineDetector
    ) -> None:
        with pytest.raises(ValueError, match="sentence-capable"):
            analyze_document("Some text.", pipeline=pipeline, doc_baseline=doc_baseline)

    def test_rejects_empty_document(
        self, pipeline: FeaturePipeline, sentence_baseline: BaselineDetector
    ) -> None:
        with pytest.raises(ValueError, match="empty"):
            analyze_document("   ", pipeline=pipeline, sentence_baseline=sentence_baseline)
