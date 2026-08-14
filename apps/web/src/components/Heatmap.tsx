import { useState } from "react";
import type { AnalysisResult, SentenceHeat, SentenceStatistics } from "../api/types";
import { EvidenceBadge } from "./Badge";

/**
 * Background color for a sentence: blue tint for human-leaning, orange tint
 * for AI-leaning, near-transparent around 0.5. Color is never the only
 * encoding — each sentence also carries its numeric P(ai) in its accessible
 * label and (for AI-leaning) a dotted underline.
 */
export function heatColor(pAi: number): string {
  if (pAi >= 0.5) {
    const alpha = (pAi - 0.5) * 2 * 0.5;
    return `rgb(239 125 90 / ${alpha.toFixed(3)})`;
  }
  const alpha = (0.5 - pAi) * 2 * 0.42;
  return `rgb(90 162 240 / ${alpha.toFixed(3)})`;
}

interface HeatmapProps {
  analysis: AnalysisResult;
}

function findStats(
  statistics: SentenceStatistics[],
  sentence: SentenceHeat,
): SentenceStatistics | undefined {
  return statistics.find((s) => s.start === sentence.start && s.end === sentence.end);
}

function SentenceDetail({
  sentence,
  stats,
}: {
  sentence: SentenceHeat;
  stats: SentenceStatistics | undefined;
}) {
  return (
    <div className="sentence-detail" data-testid="sentence-detail">
      <p className="sentence-detail__text">“{sentence.text}”</p>
      <div className="kv-row">
        <div className="kv">
          <div className="kv__key">
            P(ai) <EvidenceBadge kind="model" />
          </div>
          {(sentence.p_ai * 100).toFixed(1)}%
        </div>
        {stats && (
          <>
            <div className="kv">
              <div className="kv__key">words</div>
              {stats.n_words}
            </div>
            {stats.perplexity !== null && (
              <div className="kv">
                <div className="kv__key">
                  perplexity <EvidenceBadge kind="heuristic" />
                </div>
                {stats.perplexity.toFixed(2)}
              </div>
            )}
            {stats.mean_surprisal_bits !== null && (
              <div className="kv">
                <div className="kv__key">
                  mean surprisal <EvidenceBadge kind="heuristic" />
                </div>
                {stats.mean_surprisal_bits.toFixed(2)} bits
              </div>
            )}
            {stats.max_surprisal_bits !== null && (
              <div className="kv">
                <div className="kv__key">max surprisal</div>
                {stats.max_surprisal_bits.toFixed(2)} bits
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

export function Heatmap({ analysis }: HeatmapProps) {
  const [selected, setSelected] = useState<number | null>(null);
  const { heatmap, sentence_statistics } = analysis.evidence;
  const selectedSentence = selected !== null ? heatmap[selected] : undefined;

  return (
    <section className="panel" aria-label="Sentence heatmap">
      <h2 className="panel__title">
        Sentence heatmap <EvidenceBadge kind="model" />
      </h2>
      <p className="panel__subtitle">
        Each sentence is shaded by its learned AI probability. Select a sentence to
        inspect its statistical evidence.
      </p>
      <div className="heatmap__doc" data-testid="heatmap-doc">
        {heatmap.map((sentence, index) => (
          <span key={sentence.start}>
            {index > 0 ? " " : ""}
            <button
              type="button"
              className={[
                "sent",
                sentence.p_ai >= 0.5 ? "sent--ai-leaning" : "",
                selected === index ? "sent--selected" : "",
              ]
                .filter(Boolean)
                .join(" ")}
              style={{ backgroundColor: heatColor(sentence.p_ai) }}
              aria-label={`Sentence ${index + 1}: AI probability ${(sentence.p_ai * 100).toFixed(0)}%`}
              aria-pressed={selected === index}
              onClick={() => setSelected(selected === index ? null : index)}
            >
              {sentence.text}
            </button>
          </span>
        ))}
      </div>
      <div className="legend" aria-hidden="true">
        <span>human-leaning</span>
        <span className="legend__bar" />
        <span>AI-leaning</span>
        <span style={{ marginLeft: 12 }}>· dotted underline = P(ai) ≥ 50%</span>
      </div>
      {selectedSentence && (
        <SentenceDetail
          sentence={selectedSentence}
          stats={findStats(sentence_statistics, selectedSentence)}
        />
      )}
    </section>
  );
}
