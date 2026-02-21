from __future__ import annotations

import math
from typing import Optional
import torch


class CosineLRScheduler:
    """
    Simple cosine decay with optional warmup.
    """
    def __init__(self, optimizer: torch.optim.Optimizer, max_epochs: int, warmup_epochs: int = 0, min_lr_ratio: float = 0.05):
        self.opt = optimizer
        self.max_epochs = max_epochs
        self.warmup = warmup_epochs
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]

    def step(self, epoch: int) -> None:
        if epoch < self.warmup and self.warmup > 0:
            t = (epoch + 1) / float(self.warmup)
            factor = t
        else:
            t = (epoch - self.warmup) / float(max(1, self.max_epochs - self.warmup))
            factor = self.min_lr_ratio + 0.5 * (1.0 - self.min_lr_ratio) * (1.0 + math.cos(math.pi * t))

        for lr0, g in zip(self.base_lrs, self.opt.param_groups):
            g["lr"] = lr0 * factor