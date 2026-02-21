from __future__ import annotations

from typing import Optional

import torch


def resolve_device(device: str = "cuda") -> torch.device:
    """
    Robust device resolver:
      - 'cuda' uses cuda if available else cpu
      - 'cuda:0' etc supported
      - 'cpu' supported
    """
    device = device.strip().lower()
    if device.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(device)
        return torch.device("cpu")
    if device == "cpu":
        return torch.device("cpu")
    # fallback
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")