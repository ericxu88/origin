"""Perplexity / surprisal / entropy features (SPEC F-1..F-7).

Definitions (see :mod:`origin_ml.scoring.base` for unit conventions):

- document perplexity: ``exp(-mean(logprob))`` over all scored tokens.
- token surprisal (bits): ``-logprob / ln 2``.
- sentence perplexity: ``exp(-mean(logprob))`` over the tokens assigned to the
  sentence span (token-midpoint rule); sentences with no tokens are skipped.
- burstiness: Goh–Barabási index over the sentence-perplexity sequence.
- entropy features use the scorer's next-token distribution entropy (nats)
  when supported; otherwise they are 0.0 and ``ppl.has_entropy`` is 0.0
  (graceful degradation, SPEC F-7).
"""

from __future__ import annotations

from origin_ml.features._stats import (
    burstiness,
    coeff_variation,
    safe_max,
    safe_mean,
    safe_min,
    safe_pstd,
)
from origin_ml.features.base import AnalyzedText
from origin_ml.scoring.base import perplexity

__all__ = ["PerplexityExtractor"]


class PerplexityExtractor:
    """LM-based features; requires a scorer (SPEC F-1..F-7)."""

    _NAMES = (
        "ppl.doc_perplexity",
        "ppl.mean_token_logprob",
        "ppl.mean_surprisal_bits",
        "ppl.std_surprisal_bits",
        "ppl.max_surprisal_bits",
        "ppl.sent_ppl_mean",
        "ppl.sent_ppl_std",
        "ppl.sent_ppl_var",
        "ppl.sent_ppl_range",
        "ppl.sent_ppl_min",
        "ppl.sent_ppl_max",
        "ppl.sent_ppl_cv",
        "ppl.sent_ppl_burstiness",
        "ppl.entropy_mean",
        "ppl.entropy_std",
        "ppl.has_entropy",
    )

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._NAMES

    @property
    def requires_scoring(self) -> bool:
        return True

    def extract(self, doc: AnalyzedText) -> dict[str, float]:
        if doc.scored is None:
            raise ValueError("PerplexityExtractor requires a scored document")
        scored = doc.scored
        logprobs = scored.logprobs
        surprisals = [t.surprisal_bits for t in scored.tokens]

        sent_ppls: list[float] = []
        for sentence in doc.sentences:
            sent_tokens = scored.tokens_in_span(sentence.start, sentence.end)
            if sent_tokens:
                sent_ppls.append(perplexity([t.logprob for t in sent_tokens]))

        entropies = [t.entropy for t in scored.tokens if t.entropy is not None]
        has_entropy = 1.0 if entropies else 0.0

        return {
            "ppl.doc_perplexity": perplexity(logprobs),
            "ppl.mean_token_logprob": safe_mean(logprobs),
            "ppl.mean_surprisal_bits": safe_mean(surprisals),
            "ppl.std_surprisal_bits": safe_pstd(surprisals),
            "ppl.max_surprisal_bits": safe_max(surprisals),
            "ppl.sent_ppl_mean": safe_mean(sent_ppls),
            "ppl.sent_ppl_std": safe_pstd(sent_ppls),
            "ppl.sent_ppl_var": safe_pstd(sent_ppls) ** 2,
            "ppl.sent_ppl_range": safe_max(sent_ppls) - safe_min(sent_ppls),
            "ppl.sent_ppl_min": safe_min(sent_ppls),
            "ppl.sent_ppl_max": safe_max(sent_ppls),
            "ppl.sent_ppl_cv": coeff_variation(sent_ppls),
            "ppl.sent_ppl_burstiness": burstiness(sent_ppls),
            "ppl.entropy_mean": safe_mean(entropies),
            "ppl.entropy_std": safe_pstd(entropies),
            "ppl.has_entropy": has_entropy,
        }
