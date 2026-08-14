import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { Heatmap, heatColor } from "../components/Heatmap";
import { ResearchPanel } from "../components/ResearchPanel";
import { MIXED_RESPONSE } from "./fixtures";

const analysis = MIXED_RESPONSE.analysis;

describe("Heatmap (SPEC W-2, W-3)", () => {
  it("renders every sentence with an accessible probability label", () => {
    render(<Heatmap analysis={analysis} />);
    const sentences = screen.getAllByRole("button", { name: /AI probability/ });
    expect(sentences).toHaveLength(3);
    expect(sentences[0]).toHaveTextContent("The crooked lantern flickered.");
    expect(sentences[0]).toHaveAccessibleName(/8%/);
  });

  it("marks AI-leaning sentences with a non-color cue", () => {
    render(<Heatmap analysis={analysis} />);
    const sentences = screen.getAllByRole("button", { name: /AI probability/ });
    expect(sentences[1]?.className).toContain("sent--ai-leaning");
    expect(sentences[0]?.className).not.toContain("sent--ai-leaning");
  });

  it("opens the sentence detail with statistical evidence on click", async () => {
    render(<Heatmap analysis={analysis} />);
    const sentences = screen.getAllByRole("button", { name: /AI probability/ });
    expect(screen.queryByTestId("sentence-detail")).not.toBeInTheDocument();

    await userEvent.click(sentences[1] as HTMLElement);
    const detail = screen.getByTestId("sentence-detail");
    expect(detail).toHaveTextContent("93.0%");
    expect(detail).toHaveTextContent("6.3"); // sentence perplexity
    expect(detail).toHaveTextContent("bits");

    // Clicking again deselects (toggle, not a dead control).
    await userEvent.click(sentences[1] as HTMLElement);
    expect(screen.queryByTestId("sentence-detail")).not.toBeInTheDocument();
  });

  it("heatColor maps probability to distinct human/AI tints", () => {
    expect(heatColor(0.95)).toContain("239 125 90");
    expect(heatColor(0.05)).toContain("90 162 240");
  });
});

describe("ResearchPanel (SPEC W-4)", () => {
  it("shows the surprisal chart by default with heuristic badge", () => {
    render(<ResearchPanel analysis={analysis} />);
    expect(screen.getByRole("img", { name: /token surprisal chart/i })).toBeInTheDocument();
    expect(screen.getByText(/stub\(seed=0/)).toBeInTheDocument();
  });

  it("switches to the document feature table", async () => {
    render(<ResearchPanel analysis={analysis} />);
    await userEvent.click(screen.getByRole("tab", { name: "Document features" }));
    const table = screen.getByTestId("feature-table");
    expect(table).toHaveTextContent("ppl.doc_perplexity");
    expect(table).toHaveTextContent("14.2100");
  });

  it("shows the observed-vs-expected comparison with closer-to badges", async () => {
    render(<ResearchPanel analysis={analysis} />);
    await userEvent.click(screen.getByRole("tab", { name: "vs. expected" }));
    const table = screen.getByTestId("comparison-table");
    expect(table).toHaveTextContent("human μ±σ");
    expect(table).toHaveTextContent("19.500±3.200");
  });

  it("explains when no distribution comparison is available", async () => {
    const withoutComparison = {
      ...analysis,
      evidence: { ...analysis.evidence, distribution_comparison: null },
    };
    render(<ResearchPanel analysis={withoutComparison} />);
    await userEvent.click(screen.getByRole("tab", { name: "vs. expected" }));
    expect(screen.getByTestId("no-comparison")).toHaveTextContent(/does not embed/);
  });

  it("explains when no surprisal series is available", () => {
    const withoutSurprisal = {
      ...analysis,
      evidence: { ...analysis.evidence, token_surprisals: null },
    };
    render(<ResearchPanel analysis={withoutSurprisal} />);
    expect(screen.getByTestId("no-surprisal")).toHaveTextContent(/without a language-model/);
  });
});
