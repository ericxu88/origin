"""Hugging Face causal-LM scorer (SPEC F-1, AD-5).

Scores each token's log probability under a configurable causal language model
(any local path or hub checkpoint, e.g. ``distilgpt2``), with optional full
next-token-distribution entropy (SPEC F-7).

Conventions:

- The first token of a text has no left context under a causal LM and is
  therefore not scored (standard practice; documented rather than faked).
- Special tokens and zero-width offsets are skipped.
- Inputs longer than ``max_length`` tokens are truncated for scoring.
- Runs on CPU by default; uses CUDA/MPS automatically when available (N-5).
"""

from __future__ import annotations

import torch

from origin_ml.device import resolve_device
from origin_ml.scoring.base import ScoredText, ScoredToken

__all__ = ["HFCausalScorer"]


class HFCausalScorer:
    """Token log probabilities and entropies from a causal LM."""

    def __init__(
        self,
        checkpoint: str = "distilgpt2",
        *,
        device: str | None = None,
        max_length: int = 512,
        compute_entropy: bool = True,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._checkpoint = checkpoint
        self._max_length = max_length
        self._compute_entropy = compute_entropy
        self._device = resolve_device(device)
        self._tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForCausalLM.from_pretrained(checkpoint)
        self._model = model.to(self._device).eval()

    @property
    def name(self) -> str:
        return f"hf-causal({self._checkpoint})"

    @property
    def supports_entropy(self) -> bool:
        return self._compute_entropy

    @property
    def device(self) -> torch.device:
        return self._device

    def score(self, text: str) -> ScoredText:
        if not text.strip():
            return ScoredText(text=text, tokens=(), scorer_name=self.name)

        encoding = self._tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self._max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"]
        offsets = encoding["offset_mapping"][0].tolist()
        n_tokens = input_ids.shape[1]
        if n_tokens < 2:
            # A single token has no conditional probability under a causal LM.
            return ScoredText(text=text, tokens=(), scorer_name=self.name)

        with torch.no_grad():
            logits = self._model(input_ids.to(self._device)).logits[0].to("cpu")
        log_probs = torch.log_softmax(logits.float(), dim=-1)

        entropies: torch.Tensor | None = None
        if self._compute_entropy:
            entropies = -(log_probs.exp() * log_probs).sum(dim=-1)

        tokens: list[ScoredToken] = []
        ids = input_ids[0]
        for i in range(1, n_tokens):
            start, end = offsets[i]
            if end <= start:
                continue  # special/zero-width token
            tokens.append(
                ScoredToken(
                    text=text[start:end],
                    start=int(start),
                    end=int(end),
                    logprob=float(log_probs[i - 1, ids[i]]),
                    entropy=float(entropies[i - 1]) if entropies is not None else None,
                )
            )
        return ScoredText(text=text, tokens=tuple(tokens), scorer_name=self.name)
