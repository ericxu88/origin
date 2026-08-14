import { useCallback, useEffect, useState } from "react";
import { analyzeText, ApiError, fetchDetectors } from "./api/client";
import type { AnalyzeResponse, DetectorInfo } from "./api/types";
import { Heatmap } from "./components/Heatmap";
import { ResearchPanel } from "./components/ResearchPanel";
import { VerdictCard } from "./components/VerdictCard";

const EXAMPLE_TEXT =
  "The crooked lantern flickered near the abandoned mill, and nobody slept that night. " +
  "Rain kept falling until dawn; the dogs refused to settle. " +
  "Additionally, the system provides a clear and effective way to improve overall results. " +
  "Moreover, the process helps ensure that key goals are met in a consistent manner. " +
  "Furthermore, understanding the framework is essential for achieving reliable performance. " +
  "Who remembers the orchard now? It was the ledger, though nobody could say why.";

type Status =
  | { state: "idle" }
  | { state: "loading" }
  | { state: "error"; message: string }
  | { state: "done"; response: AnalyzeResponse };

export default function App() {
  const [text, setText] = useState("");
  const [status, setStatus] = useState<Status>({ state: "idle" });
  const [detectors, setDetectors] = useState<DetectorInfo[]>([]);
  const [detector, setDetector] = useState<string>("");

  useEffect(() => {
    let cancelled = false;
    fetchDetectors()
      .then((response) => {
        if (!cancelled) {
          setDetectors(response.detectors);
          setDetector(response.default);
        }
      })
      .catch(() => {
        // The backend may not be running yet; analysis will surface the error.
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const runAnalysis = useCallback(async () => {
    setStatus({ state: "loading" });
    try {
      const response = await analyzeText(text, detector || undefined);
      setStatus({ state: "done", response });
    } catch (error) {
      const message =
        error instanceof ApiError
          ? error.message
          : "Could not reach the Origin API. Is the backend running on port 8000?";
      setStatus({ state: "error", message });
    }
  }, [text, detector]);

  const canAnalyze = text.trim().length > 0 && status.state !== "loading";

  return (
    <div className="app">
      <header className="masthead">
        <h1 className="masthead__wordmark">origin</h1>
        <span className="masthead__tagline">
          interpretable LLM-text detection, localization &amp; evidence
        </span>
        <span className="masthead__status">
          {detectors.length > 0
            ? `${detectors.length} detector${detectors.length > 1 ? "s" : ""} loaded`
            : "connecting to API…"}
        </span>
      </header>

      <main className="layout">
        <div>
          <section className="panel" aria-label="Document input">
            <h2 className="panel__title">Document</h2>
            <p className="panel__subtitle">
              Paste text to analyze. Longer passages (3+ sentences) give more reliable
              evidence.
            </p>
            <label className="field-label" htmlFor="doc-input" style={{ display: "none" }}>
              Document text
            </label>
            <textarea
              id="doc-input"
              className="input__textarea"
              placeholder="Paste a document here…"
              value={text}
              onChange={(event) => setText(event.target.value)}
            />
            <div className="input__row">
              <button
                type="button"
                className="button button--primary"
                disabled={!canAnalyze}
                onClick={runAnalysis}
              >
                {status.state === "loading" ? (
                  <>
                    <span className="spinner" aria-hidden="true" />
                    Analyzing…
                  </>
                ) : (
                  "Analyze"
                )}
              </button>
              <button
                type="button"
                className="button"
                onClick={() => setText(EXAMPLE_TEXT)}
              >
                Load example
              </button>
              {detectors.length > 1 && (
                <>
                  <label className="field-label" htmlFor="detector-select">
                    detector
                  </label>
                  <select
                    id="detector-select"
                    className="select"
                    value={detector}
                    onChange={(event) => setDetector(event.target.value)}
                  >
                    {detectors.map((info) => (
                      <option key={info.name} value={info.name}>
                        {info.name}
                      </option>
                    ))}
                  </select>
                </>
              )}
              <span className="input__meta">{text.length.toLocaleString()} chars</span>
            </div>
          </section>
        </div>

        <div aria-live="polite">
          {status.state === "idle" && (
            <section className="panel">
              <div className="state" data-testid="empty-state">
                Paste a document and press <strong>Analyze</strong> to see the
                Human&nbsp;/&nbsp;AI&nbsp;/&nbsp;Mixed verdict, per-sentence heatmap, and
                the statistical evidence behind it.
              </div>
            </section>
          )}
          {status.state === "loading" && (
            <section className="panel">
              <div className="state" data-testid="loading-state">
                <span className="spinner" aria-hidden="true" />
                Analyzing document…
              </div>
            </section>
          )}
          {status.state === "error" && (
            <section className="panel">
              <div className="state--error" role="alert" data-testid="error-state">
                {status.message}
              </div>
            </section>
          )}
          {status.state === "done" && (
            <>
              <VerdictCard analysis={status.response.analysis} />
              <Heatmap analysis={status.response.analysis} />
              <ResearchPanel analysis={status.response.analysis} />
            </>
          )}
        </div>
      </main>
    </div>
  );
}
