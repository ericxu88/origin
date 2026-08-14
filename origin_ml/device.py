"""Torch device resolution (SPEC N-5): GPU when available, CPU always works."""

from __future__ import annotations

import torch

__all__ = ["resolve_device"]


def resolve_device(preference: str | None = None) -> torch.device:
    """Resolve a torch device.

    Explicit ``preference`` (e.g. ``"cpu"``, ``"cuda:0"``, ``"mps"``) wins;
    otherwise pick CUDA, then Apple MPS, then CPU. Nothing in Origin requires
    a GPU — acceleration is opportunistic.
    """
    if preference is not None:
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
