"""API request/response schemas (SPEC API-4, API-5).

Every payload is a fully-typed pydantic model — including the nested evidence
bundle, which is reused directly from ``origin_ml.explainability`` so the
OpenAPI document describes the complete response shape.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from origin_ml.explainability.analyze import AnalysisResult

__all__ = [
    "AnalyzeRequest",
    "AnalyzeResponse",
    "DetectorInfo",
    "DetectorsResponse",
    "HealthResponse",
]

MAX_TEXT_CHARS = 50_000


class AnalyzeRequest(BaseModel):
    """Analyze one document."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(
        min_length=1,
        max_length=MAX_TEXT_CHARS,
        description=f"Document text (1..{MAX_TEXT_CHARS} characters, not all whitespace).",
    )
    detector: str | None = Field(
        default=None,
        description="Detector name from GET /detectors; omit for the default detector.",
    )

    @field_validator("text")
    @classmethod
    def _not_only_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must contain non-whitespace characters")
        return value


class DetectorInfo(BaseModel):
    """Metadata about one loaded detector (SPEC API-3)."""

    model_config = ConfigDict(frozen=True)

    name: str
    kind: Literal["classical", "neural"]
    description: str
    scorer: str | None
    source: str = Field(description="Where the detector came from, e.g. artifact path or 'demo'.")
    feature_names: tuple[str, ...] = ()
    training_meta: dict[str, str] = Field(default_factory=dict)


class AnalyzeResponse(BaseModel):
    """Document + sentence predictions with the tagged evidence bundle."""

    model_config = ConfigDict(frozen=True)

    analysis: AnalysisResult
    detector: DetectorInfo


class DetectorsResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    default: str
    detectors: tuple[DetectorInfo, ...]


class HealthResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: Literal["ok"]
    version: str
    detectors_loaded: tuple[str, ...]
