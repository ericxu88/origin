import type { AnalyzeResponse, DetectorsResponse } from "./types";

/** Raised for any non-2xx API response, carrying the backend's detail. */
export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
  }
}

const BASE = "/api";

async function parseError(response: Response): Promise<never> {
  let detail = `${response.status} ${response.statusText}`;
  try {
    const body: unknown = await response.json();
    if (body && typeof body === "object" && "detail" in body) {
      const raw: unknown = body.detail;
      detail = typeof raw === "string" ? raw : JSON.stringify(raw);
    }
  } catch {
    // keep the status-line fallback
  }
  throw new ApiError(response.status, detail);
}

export async function analyzeText(
  text: string,
  detector?: string,
): Promise<AnalyzeResponse> {
  const response = await fetch(`${BASE}/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(detector ? { text, detector } : { text }),
  });
  if (!response.ok) {
    await parseError(response);
  }
  return (await response.json()) as AnalyzeResponse;
}

export async function fetchDetectors(): Promise<DetectorsResponse> {
  const response = await fetch(`${BASE}/detectors`);
  if (!response.ok) {
    await parseError(response);
  }
  return (await response.json()) as DetectorsResponse;
}
