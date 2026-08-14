import { useState } from "react";
import type { AnalysisResult } from "../api/types";
import { CloserToBadge, EvidenceBadge } from "./Badge";
import { SurprisalChart } from "./SurprisalChart";

type Tab = "surprisal" | "features" | "comparison";

function FeatureTable({ features }: { features: Record<string, number> }) {
  const entries = Object.entries(features);
  return (
    <div className="data-table--scroll">
      <table className="data-table" data-testid="feature-table">
        <thead>
          <tr>
            <th scope="col">feature</th>
            <th scope="col" className="num">
              value
            </th>
          </tr>
        </thead>
        <tbody>
          {entries.map(([name, value]) => (
            <tr key={name}>
              <td>{name}</td>
              <td className="num">{value.toFixed(4)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ComparisonTable({ analysis }: { analysis: AnalysisResult }) {
  const comparison = analysis.evidence.distribution_comparison;
  if (!comparison) {
    return (
      <p className="note" data-testid="no-comparison">
        The active detector artifact does not embed training-time feature
        distributions, so no observed-vs-expected comparison is available.
      </p>
    );
  }
  return (
    <div className="data-table--scroll">
      <table className="data-table" data-testid="comparison-table">
        <thead>
          <tr>
            <th scope="col">feature</th>
            <th scope="col" className="num">
              observed
            </th>
            <th scope="col" className="num">
              human μ±σ
            </th>
            <th scope="col" className="num">
              ai μ±σ
            </th>
            <th scope="col">closer to</th>
          </tr>
        </thead>
        <tbody>
          {comparison.comparisons.map((row) => (
            <tr key={row.feature}>
              <td>{row.feature}</td>
              <td className="num">{row.observed.toFixed(3)}</td>
              <td className="num">
                {row.human_mean.toFixed(3)}±{row.human_std.toFixed(3)}
              </td>
              <td className="num">
                {row.ai_mean.toFixed(3)}±{row.ai_std.toFixed(3)}
              </td>
              <td>
                <CloserToBadge value={row.closer_to} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Advanced/research view: raw features, distributions, surprisal series. */
export function ResearchPanel({ analysis }: { analysis: AnalysisResult }) {
  const [tab, setTab] = useState<Tab>("surprisal");
  const surprisals = analysis.evidence.token_surprisals;

  return (
    <section className="panel" aria-label="Research panel">
      <h2 className="panel__title">
        Research panel <EvidenceBadge kind="heuristic" />
      </h2>
      <p className="panel__subtitle">
        Raw statistical evidence behind the prediction. Heuristic measurements are
        direct computations; only the heatmap and P(ai) values are learned-model
        outputs.
      </p>
      <div className="tabs" role="tablist" aria-label="Evidence views">
        {(
          [
            ["surprisal", "Token surprisal"],
            ["features", "Document features"],
            ["comparison", "vs. expected"],
          ] as const
        ).map(([id, label]) => (
          <button
            key={id}
            role="tab"
            type="button"
            aria-selected={tab === id}
            className={`tabs__tab ${tab === id ? "tabs__tab--active" : ""}`}
            onClick={() => setTab(id)}
          >
            {label}
          </button>
        ))}
      </div>
      {tab === "surprisal" &&
        (surprisals ? (
          <SurprisalChart series={surprisals} />
        ) : (
          <p className="note" data-testid="no-surprisal">
            The active detector runs without a language-model scorer, so no token
            surprisal series is available.
          </p>
        ))}
      {tab === "features" && (
        <FeatureTable features={analysis.evidence.document_features.features} />
      )}
      {tab === "comparison" && <ComparisonTable analysis={analysis} />}
    </section>
  );
}
