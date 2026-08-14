"""Origin API: FastAPI application (SPEC §6).

Routes are thin; all model concerns live in :mod:`origin_api.registry`
(SPEC API-6). Build the app with :func:`create_app` — tests inject their own
registry; production builds one from the environment during lifespan startup.

Run locally::

    uv run uvicorn origin_api.main:app --reload --port 8000
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

import origin_api
from origin_api.registry import DetectorRegistry
from origin_api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    DetectorsResponse,
    HealthResponse,
)

__all__ = ["app", "create_app"]

_DESCRIPTION = (
    "Interpretable detection, localization, and explanation of LLM-generated text. "
    "All results are statistical evidence with calibrated probabilities — never "
    "proof of authorship."
)


def create_app(registry: DetectorRegistry | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.registry = registry if registry is not None else DetectorRegistry.from_env()
        yield

    app = FastAPI(
        title="Origin API",
        version=origin_api.__version__,
        description=_DESCRIPTION,
        lifespan=lifespan,
    )

    cors_origins = os.environ.get(
        "ORIGIN_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[origin.strip() for origin in cors_origins if origin.strip()],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    def get_registry(request: Request) -> DetectorRegistry:
        loaded: DetectorRegistry = request.app.state.registry
        return loaded

    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    def health(request: Request) -> HealthResponse:
        """Liveness plus which detectors are loaded (SPEC API-1)."""
        return HealthResponse(
            status="ok",
            version=origin_api.__version__,
            detectors_loaded=get_registry(request).names(),
        )

    @app.get("/detectors", response_model=DetectorsResponse, tags=["meta"])
    def detectors(request: Request) -> DetectorsResponse:
        """Available detectors and their metadata (SPEC API-3)."""
        loaded = get_registry(request)
        return DetectorsResponse(default=loaded.default_name, detectors=loaded.infos())

    @app.post(
        "/analyze",
        response_model=AnalyzeResponse,
        responses={404: {"description": "Unknown detector name"}},
        tags=["analysis"],
    )
    def analyze(payload: AnalyzeRequest, request: Request) -> AnalyzeResponse:
        """Analyze a document: verdict, sentence heatmap, evidence (SPEC API-2)."""
        loaded = get_registry(request)
        try:
            detector = loaded.get(payload.detector or loaded.default_name)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        try:
            result = detector.analyze(payload.text)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return AnalyzeResponse(analysis=result, detector=detector.info())

    return app


app = create_app()
