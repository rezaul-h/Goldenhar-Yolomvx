from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class AverageMeter:
    """
    Tracks running average of a scalar.
    """
    name: str
    val: float = 0.0
    avg: float = 0.0
    sum: float = 0.0
    count: int = 0

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, v: float, n: int = 1) -> None:
        self.val = float(v)
        self.sum += float(v) * int(n)
        self.count += int(n)
        self.avg = self.sum / max(1, self.count)


@dataclass
class BestTracker:
    """
    Tracks best value and epoch, for min or max criteria.
    """
    mode: str = "min"  # "min" or "max"
    best: Optional[float] = None
    best_epoch: Optional[int] = None

    def is_better(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "min":
            return value < self.best
        if self.mode == "max":
            return value > self.best
        raise ValueError("mode must be 'min' or 'max'")

    def update(self, value: float, epoch: int) -> bool:
        value = float(value)
        if self.is_better(value):
            self.best = value
            self.best_epoch = int(epoch)
            return True
        return False