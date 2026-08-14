import type { AnalyzeResponse, DetectorsResponse } from "../api/types";

const DOC =
  "The crooked lantern flickered. Additionally, the system provides clear results. Rain again.";

/** A realistic mixed-document response mirroring the backend contract. */
export const MIXED_RESPONSE: AnalyzeResponse = {
  analysis: {
    label: "mixed",
    confidence: 0.72,
    class_probabilities: { human: 0.3, ai: 0, mixed: 0.7 },
    mean_p_ai: 0.48,
    frac_ai_sentences: 0.33,
    document_p_ai: 0.41,
    detector: "baseline-logreg",
    disclaimer:
      "Origin reports statistical evidence, not proof. AI-text detection is " +
      "probabilistic and can be wrong — especially for short, heavily edited, " +
      "or non-native-English writing. Treat results as a signal for review, " +
      "never as ground truth.",
    evidence: {
      heatmap: [
        { kind: "model", text: "The crooked lantern flickered.", start: 0, end: 30, p_ai: 0.08 },
        {
          kind: "model",
          text: "Additionally, the system provides clear results.",
          start: 31,
          end: 79,
          p_ai: 0.93,
        },
        { kind: "model", text: "Rain again.", start: 80, end: 91, p_ai: 0.22 },
      ],
      sentence_statistics: [
        {
          kind: "heuristic",
          start: 0,
          end: 30,
          n_words: 4,
          n_scored_tokens: 4,
          perplexity: 21.4,
          mean_surprisal_bits: 4.42,
          max_surprisal_bits: 6.1,
        },
        {
          kind: "heuristic",
          start: 31,
          end: 79,
          n_words: 6,
          n_scored_tokens: 6,
          perplexity: 6.3,
          mean_surprisal_bits: 2.65,
          max_surprisal_bits: 3.9,
        },
        {
          kind: "heuristic",
          start: 80,
          end: 91,
          n_words: 2,
          n_scored_tokens: 2,
          perplexity: 18.8,
          mean_surprisal_bits: 4.23,
          max_surprisal_bits: 4.8,
        },
      ],
      token_surprisals: {
        kind: "heuristic",
        scorer: "stub(seed=0,bias=1.0,spread=4.0)",
        tokens: [
          { text: "The", start: 0, end: 3, logprob: -1.2, surprisal_bits: 1.73, entropy: 2.1 },
          {
            text: "crooked",
            start: 4,
            end: 11,
            logprob: -4.4,
            surprisal_bits: 6.35,
            entropy: 3.4,
          },
          {
            text: "lantern",
            start: 12,
            end: 19,
            logprob: -3.9,
            surprisal_bits: 5.63,
            entropy: 3.1,
          },
          {
            text: "Additionally",
            start: 31,
            end: 43,
            logprob: -1.4,
            surprisal_bits: 2.02,
            entropy: 1.8,
          },
        ],
      },
      document_features: {
        kind: "heuristic",
        scorer: "stub(seed=0,bias=1.0,spread=4.0)",
        features: {
          "ppl.doc_perplexity": 14.21,
          "sent.count": 3,
          "lex.ttr": 0.9231,
          "style.stopword_ratio": 0.3846,
        },
      },
      distribution_comparison: {
        kind: "heuristic",
        comparisons: [
          {
            feature: "ppl.doc_perplexity",
            observed: 14.21,
            human_mean: 19.5,
            human_std: 3.2,
            ai_mean: 8.4,
            ai_std: 1.9,
            z_vs_human: -1.65,
            z_vs_ai: 3.06,
            closer_to: "human",
          },
          {
            feature: "lex.ttr",
            observed: 0.9231,
            human_mean: 0.88,
            human_std: 0.05,
            ai_mean: 0.61,
            ai_std: 0.07,
            z_vs_human: 0.86,
            z_vs_ai: 4.47,
            closer_to: "human",
          },
        ],
      },
    },
  },
  detector: {
    name: "classical",
    kind: "classical",
    description: "Calibrated logistic regression over interpretable statistical features.",
    scorer: "stub(seed=0,bias=1.0,spread=4.0)",
    source: "demo: trained at startup on the bundled synthetic sample corpus",
    feature_names: ["ppl.doc_perplexity", "sent.count", "lex.ttr", "style.stopword_ratio"],
    training_meta: { dataset: "data/sample/documents.jsonl" },
  },
};

export const DETECTORS_RESPONSE: DetectorsResponse = {
  default: "classical",
  detectors: [MIXED_RESPONSE.detector],
};

export const DOC_TEXT = DOC;

export function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}
