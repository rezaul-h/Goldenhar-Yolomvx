from __future__ import annotations

from typing import Any, Dict, Optional

from .model import YoloMvX


def build_yolomvx(cfg: Dict[str, Any]) -> YoloMvX:
    """
    Build YOLO-MvX from a dict-like config.
    Expected keys (typical):
      - num_classes: int
      - img_size: int (optional, used by some trainers)
      - width_mult: float
      - depth_mult: float
      - in_channels: int (default 3)
      - act: str ('silu'|'relu'|'gelu')
      - stem_reparam: bool
      - strides: list[int] (default [8,16,32])
      - head: dict (optional overrides)
      - backbone: dict (optional overrides)
      - neck: dict (optional overrides)
    """
    return YoloMvX.from_config(cfg)