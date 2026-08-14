import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import App from "../App";
import { DETECTORS_RESPONSE, MIXED_RESPONSE, jsonResponse } from "./fixtures";

function urlOf(input: RequestInfo | URL): string {
  if (typeof input === "string") return input;
  return input instanceof URL ? input.href : input.url;
}

function stubFetch(
  onAnalyze: (init: RequestInit | undefined) => Response | Promise<Response>,
) {
  const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
    const url = urlOf(input);
    if (url.endsWith("/detectors")) {
      return Promise.resolve(jsonResponse(DETECTORS_RESPONSE));
    }
    if (url.endsWith("/analyze")) {
      return Promise.resolve(onAnalyze(init));
    }
    throw new Error(`unexpected fetch: ${url}`);
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

describe("App analyze flow (SPEC W-1, W-6)", () => {
  it("shows the empty state and disables Analyze until text is entered", async () => {
    stubFetch(() => jsonResponse(MIXED_RESPONSE));
    render(<App />);
    expect(screen.getByTestId("empty-state")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Analyze" })).toBeDisabled();

    await userEvent.type(screen.getByPlaceholderText(/paste a document/i), "Hello there.");
    expect(screen.getByRole("button", { name: "Analyze" })).toBeEnabled();
  });

  it("runs the full analyze flow and renders verdict, heatmap, and evidence", async () => {
    const fetchMock = stubFetch(() => jsonResponse(MIXED_RESPONSE));
    render(<App />);

    await userEvent.type(screen.getByPlaceholderText(/paste a document/i), "Some doc.");
    await userEvent.click(screen.getByRole("button", { name: "Analyze" }));

    expect(await screen.findByTestId("verdict-label")).toHaveTextContent("LIKELY MIXED");
    expect(screen.getByTestId("disclaimer")).toHaveTextContent("not proof");

    // Three-way class scores rendered from the fixture (30/70/0).
    const classScores = screen.getByTestId("class-scores");
    expect(classScores).toHaveTextContent("Human 30%");
    expect(classScores).toHaveTextContent("Mixed 70%");
    expect(classScores).toHaveTextContent("AI 0%");

    // Heatmap renders one accessible button per sentence with its probability.
    const sentences = screen.getAllByRole("button", { name: /AI probability/ });
    expect(sentences).toHaveLength(3);
    expect(sentences[1]).toHaveAccessibleName(/93%/);

    // The analyze request carried the typed text.
    const analyzeCall = fetchMock.mock.calls.find(([input]) =>
      urlOf(input).endsWith("/analyze"),
    );
    expect(analyzeCall).toBeDefined();
    const body = JSON.parse(analyzeCall?.[1]?.body as string) as { text: string };
    expect(body.text).toBe("Some doc.");
  });

  it("loads the example text via the Load example button (SPEC W-7)", async () => {
    stubFetch(() => jsonResponse(MIXED_RESPONSE));
    render(<App />);
    await userEvent.click(screen.getByRole("button", { name: "Load example" }));
    const textarea = screen.getByPlaceholderText<HTMLTextAreaElement>(/paste a document/i);
    expect(textarea.value).toMatch(/crooked lantern/);
    expect(screen.getByRole("button", { name: "Analyze" })).toBeEnabled();
  });

  it("surfaces backend error detail in the error state", async () => {
    stubFetch(() => jsonResponse({ detail: "document contains no sentences" }, 422));
    render(<App />);
    await userEvent.type(screen.getByPlaceholderText(/paste a document/i), "x");
    await userEvent.click(screen.getByRole("button", { name: "Analyze" }));

    const error = await screen.findByTestId("error-state");
    expect(error).toHaveTextContent("document contains no sentences");
  });

  it("shows a helpful message when the API is unreachable", async () => {
    const fetchMock = vi.fn((_input: RequestInfo | URL) =>
      Promise.reject(new Error("connection refused")),
    );
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);
    await userEvent.type(screen.getByPlaceholderText(/paste a document/i), "Hello.");
    await userEvent.click(screen.getByRole("button", { name: "Analyze" }));
    expect(await screen.findByTestId("error-state")).toHaveTextContent(/backend running/i);
  });

  it("reports loaded detectors in the masthead", async () => {
    stubFetch(() => jsonResponse(MIXED_RESPONSE));
    render(<App />);
    await waitFor(() =>
      expect(screen.getByText(/1 detector loaded/)).toBeInTheDocument(),
    );
  });
});
