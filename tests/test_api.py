"""API contract tests with an injected registry (SPEC API-1..API-6)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from origin_api.main import create_app
from origin_api.registry import DetectorRegistry
from origin_api.schemas import MAX_TEXT_CHARS

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample" / "documents.jsonl"

MIXED_TEXT = (
    "The crooked lantern flickered near the abandoned mill. "
    "Additionally, the system provides a clear and effective way to improve overall results. "
    "Moreover, the process helps ensure that key goals are met in a consistent manner. "
    "Rain again."
)


@pytest.fixture(scope="module")
def registry() -> DetectorRegistry:
    return DetectorRegistry.demo(sample_path=SAMPLE)


@pytest.fixture(scope="module")
def client(registry: DetectorRegistry) -> Iterator[TestClient]:
    """App with an injected demo registry (no env, no network; SPEC API-6)."""
    app = create_app(registry=registry)
    with TestClient(app) as test_client:
        yield test_client


class TestHealth:
    def test_health(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert "classical" in body["detectors_loaded"]
        assert body["version"]


class TestDetectors:
    def test_metadata(self, client: TestClient) -> None:
        response = client.get("/detectors")
        assert response.status_code == 200
        body = response.json()
        assert body["default"] == "classical"
        (classical,) = [d for d in body["detectors"] if d["name"] == "classical"]
        assert classical["kind"] == "classical"
        assert classical["scorer"] is not None
        assert "demo" in classical["source"]
        assert len(classical["feature_names"]) > 20


class TestAnalyze:
    def test_full_response_contract(self, client: TestClient) -> None:
        response = client.post("/analyze", json={"text": MIXED_TEXT})
        assert response.status_code == 200
        body = response.json()

        analysis = body["analysis"]
        assert analysis["label"] in ("human", "ai", "mixed")
        assert 0.0 <= analysis["confidence"] <= 1.0
        assert "not proof" in analysis["disclaimer"]

        heatmap = analysis["evidence"]["heatmap"]
        assert len(heatmap) == 4
        for sentence in heatmap:
            assert MIXED_TEXT[sentence["start"] : sentence["end"]] == sentence["text"]
            assert sentence["kind"] == "model"
            assert 0.0 <= sentence["p_ai"] <= 1.0

        stats = analysis["evidence"]["sentence_statistics"]
        assert len(stats) == 4
        assert all(s["kind"] == "heuristic" for s in stats)

        surprisals = analysis["evidence"]["token_surprisals"]
        assert surprisals is not None and surprisals["kind"] == "heuristic"
        assert surprisals["tokens"]

        comparison = analysis["evidence"]["distribution_comparison"]
        assert comparison is not None and comparison["comparisons"]

        scores = analysis["class_probabilities"]
        assert set(scores) == {"human", "ai", "mixed"}
        assert abs(sum(scores.values()) - 1.0) < 1e-9

        assert body["detector"]["name"] == "classical"

    def test_localization_direction(self, client: TestClient) -> None:
        """AI-patterned middle sentences should outscore the human-patterned ends."""
        body = client.post("/analyze", json={"text": MIXED_TEXT}).json()
        probs = [s["p_ai"] for s in body["analysis"]["evidence"]["heatmap"]]
        assert (probs[1] + probs[2]) / 2 > (probs[0] + probs[3]) / 2

    def test_explicit_detector_choice(self, client: TestClient) -> None:
        response = client.post("/analyze", json={"text": "Hello there.", "detector": "classical"})
        assert response.status_code == 200

    def test_unknown_detector_404(self, client: TestClient) -> None:
        response = client.post("/analyze", json={"text": "Hello there.", "detector": "bogus"})
        assert response.status_code == 404
        assert "available: classical" in response.json()["detail"]

    def test_empty_text_422(self, client: TestClient) -> None:
        assert client.post("/analyze", json={"text": ""}).status_code == 422

    def test_whitespace_text_422(self, client: TestClient) -> None:
        response = client.post("/analyze", json={"text": " \n\t "})
        assert response.status_code == 422
        assert "non-whitespace" in str(response.json()["detail"])

    def test_oversize_text_422(self, client: TestClient) -> None:
        response = client.post("/analyze", json={"text": "a" * (MAX_TEXT_CHARS + 1)})
        assert response.status_code == 422

    def test_missing_body_422(self, client: TestClient) -> None:
        assert client.post("/analyze", json={}).status_code == 422


class TestOpenAPI:
    def test_full_schema_modeled(self, client: TestClient) -> None:
        """SPEC API-5: nested payloads are real models, not dict blobs."""
        schemas = client.get("/openapi.json").json()["components"]["schemas"]
        for name in (
            "AnalyzeRequest",
            "AnalyzeResponse",
            "AnalysisResult",
            "EvidenceBundle",
            "SentenceHeat",
            "TokenSurprisalSeries",
            "DetectorInfo",
            "HealthResponse",
        ):
            assert name in schemas, f"missing OpenAPI schema {name}"


class TestRegistry:
    def test_unknown_default_rejected(self) -> None:
        with pytest.raises(ValueError, match="default detector"):
            DetectorRegistry({}, default="classical")

    def test_get_unknown_detector_message(self, registry: DetectorRegistry) -> None:
        with pytest.raises(KeyError, match="available: classical"):
            registry.get("nope")
