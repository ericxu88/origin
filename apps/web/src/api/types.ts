/**
 * TypeScript mirrors of the Origin API pydantic schemas.
 * Source of truth: apps/api/origin_api/schemas.py + origin_ml/explainability.
 */

export type DocLabel = "human" | "ai" | "mixed";
export type EvidenceKind = "heuristic" | "model";

export interface SentenceHeat {
  kind: "model";
  text: string;
  start: number;
  end: number;
  p_ai: number;
}

export interface SentenceStatistics {
  kind: "heuristic";
  start: number;
  end: number;
  n_words: number;
  n_scored_tokens: number;
  perplexity: number | null;
  mean_surprisal_bits: number | null;
  max_surprisal_bits: number | null;
}

export interface TokenSurprisal {
  text: string;
  start: number;
  end: number;
  logprob: number;
  surprisal_bits: number;
  entropy: number | null;
}

export interface TokenSurprisalSeries {
  kind: "heuristic";
  scorer: string;
  tokens: TokenSurprisal[];
}

export interface DocumentFeatureSummary {
  kind: "heuristic";
  scorer: string | null;
  features: Record<string, number>;
}

export interface FeatureComparison {
  feature: string;
  observed: number;
  human_mean: number;
  human_std: number;
  ai_mean: number;
  ai_std: number;
  z_vs_human: number;
  z_vs_ai: number;
  closer_to: "human" | "ai" | "similar";
}

export interface DistributionComparisonSection {
  kind: "heuristic";
  comparisons: FeatureComparison[];
}

export interface EvidenceBundle {
  heatmap: SentenceHeat[];
  sentence_statistics: SentenceStatistics[];
  token_surprisals: TokenSurprisalSeries | null;
  document_features: DocumentFeatureSummary;
  distribution_comparison: DistributionComparisonSection | null;
}

export interface ClassProbabilities {
  human: number;
  ai: number;
  mixed: number;
}

export interface AnalysisResult {
  label: DocLabel;
  confidence: number;
  class_probabilities: ClassProbabilities;
  mean_p_ai: number;
  frac_ai_sentences: number;
  document_p_ai: number | null;
  detector: string;
  disclaimer: string;
  evidence: EvidenceBundle;
}

export interface DetectorInfo {
  name: string;
  kind: "classical" | "neural";
  description: string;
  scorer: string | null;
  source: string;
  feature_names: string[];
  training_meta: Record<string, string>;
}

export interface AnalyzeResponse {
  analysis: AnalysisResult;
  detector: DetectorInfo;
}

export interface DetectorsResponse {
  default: string;
  detectors: DetectorInfo[];
}

export interface HealthResponse {
  status: "ok";
  version: string;
  detectors_loaded: string[];
}
