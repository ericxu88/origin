"""Deterministic scorers for tests and offline operation.

These are *real* implementations of the :class:`~origin_ml.scoring.base.Scorer`
interface — not mocks. They exist so the entire feature/detector/API stack can
run deterministically with no model downloads (SPEC Q-2, N-6).
"""

from __future__ import annotations

import hashlib

from origin_ml.scoring.base import ScoredText, ScoredToken
from origin_ml.text.words import tokenize_words

__all__ = ["FixedScorer", "StubScorer"]


class StubScorer:
    """Hash-based deterministic scorer.

    Each word token receives a pseudo-random but fully deterministic log
    probability derived from a BLAKE2 hash of the lowercased token and the
    scorer seed. Values are stable across processes and platforms (unlike
    Python's builtin ``hash``, which is salted per process).

    ``logprob`` is mapped into ``[-(bias + spread), -bias]``; entropy into
    ``[0.5, 4.5]`` nats. ``bias`` shifts all log probabilities, which lets
    tests construct corpora whose classes differ in perplexity.
    """

    def __init__(self, seed: int = 0, bias: float = 1.0, spread: float = 4.0) -> None:
        if spread < 0:
            raise ValueError("spread must be non-negative")
        self._seed = seed
        self._bias = bias
        self._spread = spread

    @property
    def name(self) -> str:
        return f"stub(seed={self._seed},bias={self._bias},spread={self._spread})"

    @property
    def supports_entropy(self) -> bool:
        return True

    def _unit(self, token: str, salt: bytes) -> float:
        digest = hashlib.blake2b(
            f"{self._seed}:{token.lower()}".encode(), person=salt, digest_size=8
        ).digest()
        return int.from_bytes(digest, "big") / float(2**64)

    def score(self, text: str) -> ScoredText:
        tokens = [
            ScoredToken(
                text=w.text,
                start=w.start,
                end=w.end,
                logprob=-(self._bias + self._spread * self._unit(w.text, b"logprob\x00")),
                entropy=0.5 + 4.0 * self._unit(w.text, b"entropy\x00"),
            )
            for w in tokenize_words(text)
        ]
        return ScoredText(text=text, tokens=tuple(tokens), scorer_name=self.name)


class FixedScorer:
    """Table-driven scorer: exact log probabilities per lowercased token.

    Used in tests where feature values must be hand-computable (SPEC F-13).
    """

    def __init__(
        self,
        logprobs: dict[str, float],
        default: float = -2.0,
        entropies: dict[str, float] | None = None,
    ) -> None:
        self._logprobs = {k.lower(): v for k, v in logprobs.items()}
        self._default = default
        self._entropies = {k.lower(): v for k, v in (entropies or {}).items()}

    @property
    def name(self) -> str:
        return "fixed"

    @property
    def supports_entropy(self) -> bool:
        return bool(self._entropies)

    def score(self, text: str) -> ScoredText:
        tokens = [
            ScoredToken(
                text=w.text,
                start=w.start,
                end=w.end,
                logprob=self._logprobs.get(w.text.lower(), self._default),
                entropy=self._entropies.get(w.text.lower()) if self._entropies else None,
            )
            for w in tokenize_words(text)
        ]
        return ScoredText(text=text, tokens=tuple(tokens), scorer_name=self.name)
