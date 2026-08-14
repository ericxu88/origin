import type { EvidenceKind } from "../api/types";

/** Tags every piece of evidence as heuristic (statistical) or model (learned). */
export function EvidenceBadge({ kind }: { kind: EvidenceKind }) {
  const title =
    kind === "model"
      ? "Learned-model probability (classifier output)"
      : "Heuristic statistical evidence (direct measurement)";
  return (
    <span className={`badge badge--${kind}`} title={title}>
      {kind}
    </span>
  );
}

export function CloserToBadge({ value }: { value: "human" | "ai" | "similar" }) {
  return <span className={`badge badge--${value}`}>{value}</span>;
}
