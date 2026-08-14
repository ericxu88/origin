"""Generate the tiny committed model fixtures used by offline unit tests (N-6).

Usage::

    uv run python scripts/build_test_fixtures.py

Creates, deterministically (fixed seeds):

- ``tests/fixtures/tiny_tokenizer`` — a word-level tokenizer over Origin's
  embedded common-word lists plus punctuation (no downloads, ~1 KB files).
- ``tests/fixtures/tiny_classifier`` — a 2-layer BERT-style sequence
  classifier (hidden 32) with ``id2label = {0: human, 1: ai}``.
- ``tests/fixtures/tiny_causal_lm`` — a 2-layer GPT-2-style causal LM sharing
  the same tokenizer, for HFCausalScorer tests.

These are *real* Hugging Face models (randomly initialized, ~100 KB each) so
tests exercise genuine ``from_pretrained``/``save_pretrained``/forward paths
with zero network access.
"""

from __future__ import annotations

import string
from pathlib import Path

import torch
from tokenizers import Tokenizer, normalizers, pre_tokenizers
from tokenizers.models import WordLevel
from tokenizers.processors import TemplateProcessing
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
)

from origin_ml.features.wordlists import COMMON_WORDS, STOPWORDS

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
SEED = 20260813

SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def build_tokenizer() -> PreTrainedTokenizerFast:
    words = sorted(STOPWORDS | COMMON_WORDS)
    punctuation = list(string.punctuation) + list(string.digits)
    vocab = {token: i for i, token in enumerate([*SPECIALS, *words, *punctuation])}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Lowercase()
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B [SEP]",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )


def main() -> None:
    torch.manual_seed(SEED)
    tokenizer = build_tokenizer()
    vocab_size = tokenizer.vocab_size

    tokenizer.save_pretrained(FIXTURES / "tiny_tokenizer")

    classifier_config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=128,
        num_labels=2,
        id2label={0: "human", 1: "ai"},
        label2id={"human": 0, "ai": 1},
    )
    classifier = BertForSequenceClassification(classifier_config)
    classifier.save_pretrained(FIXTURES / "tiny_classifier")
    tokenizer.save_pretrained(FIXTURES / "tiny_classifier")

    lm_config = GPT2Config(
        vocab_size=vocab_size,
        n_embd=32,
        n_layer=2,
        n_head=2,
        n_positions=128,
        bos_token_id=tokenizer.cls_token_id,
        eos_token_id=tokenizer.sep_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    lm = GPT2LMHeadModel(lm_config)
    lm.save_pretrained(FIXTURES / "tiny_causal_lm")
    tokenizer.save_pretrained(FIXTURES / "tiny_causal_lm")

    for name in ("tiny_tokenizer", "tiny_classifier", "tiny_causal_lm"):
        size = sum(f.stat().st_size for f in (FIXTURES / name).rglob("*") if f.is_file())
        print(f"{name}: {size / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
