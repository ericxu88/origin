import type { AnalysisResult } from "../api/types";
import { EvidenceBadge } from "./Badge";

const LABEL_TEXT: Record<AnalysisResult["label"], string> = {
  human: "LIKELY HUMAN",
  ai: "LIKELY AI",
  mixed: "LIKELY MIXED",
};

function pct(value: number): string {
  return `${(value * 100).toFixed(0)}%`;
}

export function VerdictCard({ analysis }: { analysis: AnalysisResult }) {
  return (
    <section className="panel" aria-label="Verdict">
      <h2 className="panel__title">Verdict</h2>
      <p className="panel__subtitle">
        Probabilistic assessment by <code>{analysis.detector}</code> — not proof of
        authorship.
      </p>
      <div className="verdict">
        <span
          className={`verdict__label verdict__label--${analysis.label}`}
          data-testid="verdict-label"
        >
          {LABEL_TEXT[analysis.label]}
        </span>
        <div className="verdict__stats">
          <div className="stat">
            <div className="stat__value">{pct(analysis.confidence)}</div>
            <div className="stat__label">
              confidence <EvidenceBadge kind="model" />
            </div>
            <div className="meter" role="presentation">
              <div
                className="meter__fill"
                style={{ width: `${analysis.confidence * 100}%` }}
              />
            </div>
          </div>
          <div className="stat">
            <div className="stat__value">{pct(analysis.mean_p_ai)}</div>
            <div className="stat__label">mean sentence P(ai)</div>
          </div>
          <div className="stat">
            <div className="stat__value">{pct(analysis.frac_ai_sentences)}</div>
            <div className="stat__label">AI-leaning sentences</div>
          </div>
          {analysis.document_p_ai !== null && (
            <div className="stat">
              <div className="stat__value">{pct(analysis.document_p_ai)}</div>
              <div className="stat__label">document P(ai)</div>
            </div>
          )}
        </div>
      </div>
      <p className="disclaimer" data-testid="disclaimer">
        {analysis.disclaimer}
      </p>
    </section>
  );
}
