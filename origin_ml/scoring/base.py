"""Language-model scoring interface (SPEC AD-5).

A :class:`Scorer` assigns each token of a text a log probability (natural log)
under some language model, plus optionally the full next-token distribution
entropy at that position. Implementations:

- :class:`origin_ml.scoring.stub.StubScorer` — deterministic, dependency-free,
  used in unit tests and as an offline fallback.
- :class:`origin_ml.scoring.stub.FixedScorer` — table-driven, for
  hand-checkable feature tests.
- ``origin_ml.scoring.hf.HFCausalScorer`` — real Hugging Face causal LM.

Unit conventions used throughout Origin:

- ``logprob`` — natural-log probability of the token given its left context.
- ``surprisal`` (bits) — ``-logprob / ln(2)``.
- ``entropy`` (nats) — Shannon entropy of the model's next-token distribution
  at the token's position, or ``None`` where the scorer cannot provide it.
- document ``perplexity`` — ``exp(-mean(logprob))`` over scored tokens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

__all__ = ["LN2", "ScoredText", "ScoredToken", "Scorer", "perplexity"]

LN2 = math.log(2.0)


@dataclass(frozen=True, slots=True)
class ScoredToken:
    """A scored token with exact character offsets into the source document."""

    text: str
    start: int
    end: int
    logprob: float
    entropy: float | None = None

    @property
    def surprisal_bits(self) -> float:
        """Token surprisal in bits: ``-log2 P(token | context)``."""
        return -self.logprob / LN2


@dataclass(frozen=True, slots=True)
class ScoredText:
    """A document's tokens scored by a language model."""

    text: str
    tokens: tuple[ScoredToken, ...]
    scorer_name: str

    @property
    def logprobs(self) -> list[float]:
        return [t.logprob for t in self.tokens]

    def tokens_in_span(self, start: int, end: int) -> list[ScoredToken]:
        """Tokens whose span midpoint lies within ``[start, end)``.

        The midpoint rule assigns each token to exactly one sentence even when
        a tokenizer straddles a boundary by a character.
        """
        return [t for t in self.tokens if start <= (t.start + t.end) / 2 < end]


def perplexity(logprobs: list[float]) -> float:
    """``exp(-mean(logprobs))``; defined as 1.0 for an empty sequence."""
    if not logprobs:
        return 1.0
    return math.exp(-sum(logprobs) / len(logprobs))


@runtime_checkable
class Scorer(Protocol):
    """Assigns per-token log probabilities (and optionally entropies) to text."""

    @property
    def name(self) -> str:
        """Stable identifier recorded in artifacts and evidence output."""
        ...

    @property
    def supports_entropy(self) -> bool:
        """Whether :attr:`ScoredToken.entropy` will be populated."""
        ...

    def score(self, text: str) -> ScoredText:
        """Score ``text``; token offsets must satisfy ``text[start:end] == token.text``."""
        ...
