import { describe, expect, it, vi } from "vitest";
import { analyzeText, ApiError, fetchDetectors } from "../api/client";
import { DETECTORS_RESPONSE, MIXED_RESPONSE, jsonResponse } from "./fixtures";

function parseBody(init: RequestInit | undefined): unknown {
  return JSON.parse(init?.body as string);
}

describe("api client", () => {
  it("posts text and detector to /api/analyze", async () => {
    const fetchMock = vi.fn((_input: RequestInfo | URL, _init?: RequestInit) =>
      Promise.resolve(jsonResponse(MIXED_RESPONSE)),
    );
    vi.stubGlobal("fetch", fetchMock);

    const result = await analyzeText("Some document.", "classical");
    expect(result.analysis.label).toBe("mixed");

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/analyze",
      expect.objectContaining({ method: "POST" }),
    );
    expect(parseBody(fetchMock.mock.calls[0]?.[1])).toEqual({
      text: "Some document.",
      detector: "classical",
    });
  });

  it("omits the detector field when not chosen", async () => {
    const fetchMock = vi.fn((_input: RequestInfo | URL, _init?: RequestInit) =>
      Promise.resolve(jsonResponse(MIXED_RESPONSE)),
    );
    vi.stubGlobal("fetch", fetchMock);
    await analyzeText("Some document.");
    expect(parseBody(fetchMock.mock.calls[0]?.[1])).toEqual({ text: "Some document." });
  });

  it("throws ApiError with backend detail on non-2xx", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() => Promise.resolve(jsonResponse({ detail: "unknown detector 'x'" }, 404))),
    );
    const error = await analyzeText("doc", "x").catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).status).toBe(404);
    expect((error as ApiError).message).toBe("unknown detector 'x'");
  });

  it("falls back to the status line when the error body is not JSON", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() =>
        Promise.resolve(new Response("boom", { status: 500, statusText: "Server Error" })),
      ),
    );
    const error = await fetchDetectors().catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).message).toContain("500");
  });

  it("fetches detector metadata", async () => {
    vi.stubGlobal("fetch", vi.fn(() => Promise.resolve(jsonResponse(DETECTORS_RESPONSE))));
    const detectors = await fetchDetectors();
    expect(detectors.default).toBe("classical");
    expect(detectors.detectors[0]?.kind).toBe("classical");
  });
});
