import { useMemo } from "react";
import type { TokenSurprisalSeries } from "../api/types";
import { EvidenceBadge } from "./Badge";

const WIDTH = 720;
const HEIGHT = 150;
const PAD = { top: 10, right: 8, bottom: 22, left: 34 };

/** Hand-rolled SVG bar chart of per-token surprisal (heuristic evidence). */
export function SurprisalChart({ series }: { series: TokenSurprisalSeries }) {
  const tokens = series.tokens;
  const maxSurprisal = useMemo(
    () => Math.max(1, ...tokens.map((t) => t.surprisal_bits)),
    [tokens],
  );

  if (tokens.length === 0) {
    return <p className="note">No scored tokens available for this document.</p>;
  }

  const plotWidth = WIDTH - PAD.left - PAD.right;
  const plotHeight = HEIGHT - PAD.top - PAD.bottom;
  const barWidth = Math.max(1, plotWidth / tokens.length - 1);
  const yTicks = [0, Math.round(maxSurprisal / 2), Math.round(maxSurprisal)];

  return (
    <div>
      <svg
        className="surprisal__svg"
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-label={`Token surprisal chart: ${tokens.length} tokens, maximum ${maxSurprisal.toFixed(1)} bits`}
      >
        {yTicks.map((tick) => {
          const y = PAD.top + plotHeight - (tick / maxSurprisal) * plotHeight;
          return (
            <g key={tick}>
              <line
                x1={PAD.left}
                x2={WIDTH - PAD.right}
                y1={y}
                y2={y}
                stroke="#232c38"
                strokeWidth="1"
              />
              <text x={PAD.left - 6} y={y + 3} textAnchor="end" fontSize="9" fill="#5d6b7c">
                {tick}
              </text>
            </g>
          );
        })}
        {tokens.map((token, index) => {
          const height = (token.surprisal_bits / maxSurprisal) * plotHeight;
          const intensity = Math.min(1, token.surprisal_bits / maxSurprisal);
          return (
            <rect
              key={`${token.start}-${index}`}
              x={PAD.left + (index / tokens.length) * plotWidth}
              y={PAD.top + plotHeight - height}
              width={barWidth}
              height={Math.max(height, 0.5)}
              fill={`rgb(239 125 90 / ${(0.25 + 0.75 * intensity).toFixed(2)})`}
            >
              <title>
                {`"${token.text}" — ${token.surprisal_bits.toFixed(2)} bits` +
                  (token.entropy !== null ? `, entropy ${token.entropy.toFixed(2)} nats` : "")}
              </title>
            </rect>
          );
        })}
        <text
          x={PAD.left}
          y={HEIGHT - 6}
          fontSize="9"
          fill="#5d6b7c"
        >{`tokens in document order (n=${tokens.length}); bar height = surprisal (bits)`}</text>
      </svg>
      <p className="surprisal__caption">
        <EvidenceBadge kind="heuristic" /> token surprisal under scorer{" "}
        <code>{series.scorer}</code>; taller bars = tokens the LM found less predictable.
      </p>
    </div>
  );
}
