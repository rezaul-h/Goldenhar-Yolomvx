from __future__ import annotations

from typing import Iterable, Tuple
import torch


def build_optimizer(
    params: Iterable[torch.nn.Parameter],
    lr: float,
    weight_decay: float,
    optim_name: str = "adamw",
) -> torch.optim.Optimizer:
    name = optim_name.lower().strip()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    raise ValueError(f"Unsupported optimizer: {optim_name}")