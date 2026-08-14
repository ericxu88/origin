"""Shared test configuration: offline, deterministic (SPEC Q-2)."""

from __future__ import annotations

import os
import random

import pytest

# Enforce offline operation regardless of how pytest was invoked; unit tests
# must never download models or touch the network.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@pytest.fixture(autouse=True)
def _deterministic_seed() -> None:
    """Reset global RNG state before every test."""
    random.seed(0)
