"""Backend smoke test: in-process health + analyze round trip (SPEC V-1).

Runs completely offline with the demo classical registry (stub scorer,
sample-corpus training at startup). Exit code 0 iff every check passes.

Usage::

    uv run python scripts/smoke_api.py
"""

from __future__ import annotations


def main() -> int:
    from fastapi.testclient import TestClient
    from origin_api.main import create_app
    from origin_api.registry import DetectorRegistry

    registry = DetectorRegistry.demo()
    app = create_app(registry=registry)

    checks: list[tuple[str, bool]] = []
    with TestClient(app) as client:
        health = client.get("/health")
        checks.append(("GET /health -> 200", health.status_code == 200))
        checks.append(
            ("health reports detectors", "classical" in health.json()["detectors_loaded"])
        )

        detectors = client.get("/detectors")
        checks.append(("GET /detectors -> 200", detectors.status_code == 200))
        checks.append(("default detector present", detectors.json()["default"] == "classical"))

        response = client.post(
            "/analyze",
            json={
                "text": (
                    "The crooked lantern flickered near the abandoned mill. "
                    "Additionally, the system provides a clear and effective way "
                    "to improve overall results."
                )
            },
        )
        checks.append(("POST /analyze -> 200", response.status_code == 200))
        if response.status_code == 200:
            body = response.json()
            analysis = body["analysis"]
            checks.append(("label present", analysis["label"] in ("human", "ai", "mixed")))
            checks.append(("heatmap non-empty", len(analysis["evidence"]["heatmap"]) == 2))
            checks.append(("disclaimer present", "not proof" in analysis["disclaimer"]))
            checks.append(
                ("evidence tagged", analysis["evidence"]["heatmap"][0]["kind"] == "model")
            )

        bad = client.post("/analyze", json={"text": "   "})
        checks.append(("whitespace text -> 422", bad.status_code == 422))
        unknown = client.post("/analyze", json={"text": "Hello there.", "detector": "nope"})
        checks.append(("unknown detector -> 404", unknown.status_code == 404))

        openapi = client.get("/openapi.json")
        checks.append(
            (
                "OpenAPI models evidence bundle",
                openapi.status_code == 200
                and "EvidenceBundle" in openapi.json().get("components", {}).get("schemas", {}),
            )
        )

    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if failed:
        print(f"smoke: {len(failed)} check(s) failed")
        return 1
    print(f"smoke: all {len(checks)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
