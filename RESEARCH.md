# Origin — Research Questions & Experiment Protocol

This document defines what Origin measures, how, and how to interpret the
outputs. The implementation lives in `origin_ml/evaluation/`; the canonical
experiment entry point is:

```bash
uv run origin experiment experiments/sample.json
# equivalently: uv run python scripts/run_experiment.py experiments/sample.json
```

## Research questions

- **RQ1 — Classification.** Is a document human-written, AI-generated, or
  mixed? Measured by document-level accuracy / F1 / AUROC on held-out pure
  documents, plus the fraction of mixed documents receiving the MIXED label.
- **RQ2 — Localization.** Which sentences are most likely AI-generated?
  Measured by sentence-level precision/recall/F1/AUROC over mixed documents
  against ground-truth span labels.
- **RQ3 — Evidence.** What statistical signals support a prediction? Origin
  surfaces every feature value, per-token surprisal, and observed-vs-expected
  class distributions (from artifact-embedded training summaries) with an
  explicit heuristic/model tag on each item.
- **RQ4 — Generalization.** How well does detection transfer to unseen model
  families? Measured by retraining with an entire family held out
  (`family_holdout_split`) and evaluating only on that family (+ held-out
  human documents).
- **RQ5 — Robustness.** How stable is detection under paraphrasing, sampling
  temperature, and mixing? Measured by the paraphrased-document slice, the
  temperature metadata recorded per document, and mixed-document metrics.
- **RQ6 — Signal overlap.** Do neural detectors learn signals similar to
  classical perplexity/surprisal features? Measured by ablation deltas and
  Pearson correlations between neural sentence probabilities and classical
  features/probabilities (`rq6_correlations` in `results.json`).

## Metric definitions

All implemented in `origin_ml/evaluation/metrics.py` (unit-tested against
hand-computed values):

| Metric | Definition |
|---|---|
| accuracy | fraction of correct threshold decisions (default τ = 0.5) |
| precision / recall / F1 | standard binary definitions, positive class = AI; 0 on empty denominators |
| AUROC | `sklearn.metrics.roc_auc_score`; `null` when only one class present |
| Brier | mean squared error of `P(ai)` against the 0/1 label |
| calibration bins | 10 equal-width reliability bins: mean predicted vs observed AI fraction |
| confusion matrix | tp / fp / tn / fn at τ |

Unit conventions: token log probabilities are natural log; surprisal is
`-log2 P` (bits); perplexity is `exp(-mean logprob)`; entropy is the full
next-token distribution entropy in nats.

## Ablation matrix (RQ3/RQ6)

| System | Features | Classifier |
|---|---|---|
| `perplexity_only` | `ppl.*` (LM logprob/surprisal/entropy statistics) | logistic regression |
| `statistical_features` | sentence-length, burstiness, lexical, repetition, style — no LM | logistic regression |
| `classical_full` | all features | logistic regression |
| `neural` | learned representation | transformer sentence classifier + aggregation |
| `combined` | — | mean of `classical_full` and `neural` probabilities |

Every classical system trains **two** models: a document-level classifier
(document probability, distribution comparisons) and a sentence-level
classifier (localization). Document-level models are never applied to single
sentences — sentence vectors lie outside their training distribution (this
was observed empirically during development and is enforced by design).

## Evaluation slices (RQ4/RQ5)

Per ablation, `run_experiment` reports:

- `seen` — pure human/AI test documents (group-hashed 80/20 split; a document
  and its paraphrase can never straddle the split — `group_id` hashing).
- `paraphrased` — paraphrase-transformed test documents only.
- `localization` — sentence-level metrics over mixed test documents.
- `mixed_doc_labels` — fraction of mixed documents labelled MIXED by the
  documented aggregation rule (`origin_ml/detectors/aggregation.py`).
- `unseen_family.<f>` — retrained from scratch without family `f`, evaluated
  on family `f` plus held-out human documents.

## Outputs

`runs/<experiment>/` contains:

- `results.json` — every metric above, plus the config, seed, and git commit
  that produced it (machine-readable, schema = pydantic models).
- `ablation_auroc.{png,svg}` — document vs localization AUROC per ablation.
- `robustness_f1.{png,svg}` — seen vs paraphrased vs unseen-family F1.
- `calibration_classical_full.{png,svg}` — reliability diagram.

## Interpreting results on the sample corpus

The bundled corpus (`data/sample/`) is a **synthetic fixture** whose class
signatures are separable by construction; near-perfect scores there validate
the pipeline, not real-world detection ability. Substantive research use
requires importing real corpora via `origin_ml/datasets/adapters.py` and an
HF scorer (`--scorer hf:<checkpoint>`), then re-running the same protocol.

## Determinism

Every stochastic step is seeded from the experiment config: splits are pure
functions of `(seed, group_id)`; classical training passes the seed to
scikit-learn; neural training seeds Python/NumPy/Torch and uses a seeded
DataLoader generator. Unit tests run fully offline (`HF_HUB_OFFLINE=1`)
against committed tiny model fixtures.
