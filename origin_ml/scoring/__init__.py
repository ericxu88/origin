"""Language-model scoring: interface, deterministic stubs, and HF implementation.

``HFCausalScorer`` is exported lazily so that importing the scoring package
(e.g. from the CLI or API with stub scorers) does not pay the torch import
cost until an HF scorer is actually requested.
"""

from typing import TYPE_CHECKING, Any

from origin_ml.scoring.base import LN2, ScoredText, ScoredToken, Scorer, perplexity
from origin_ml.scoring.stub import FixedScorer, StubScorer

if TYPE_CHECKING:
    from origin_ml.scoring.hf import HFCausalScorer

__all__ = [
    "LN2",
    "FixedScorer",
    "HFCausalScorer",
    "ScoredText",
    "ScoredToken",
    "Scorer",
    "StubScorer",
    "perplexity",
]


def __getattr__(name: str) -> Any:
    if name == "HFCausalScorer":
        from origin_ml.scoring.hf import HFCausalScorer

        return HFCausalScorer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
