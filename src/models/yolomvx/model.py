from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .backbone import YoloMvXBackbone
from .neck import PANFPN
from .head import YoloHead, HeadOut


@dataclass(frozen=True)
class YoloMvXConfig:
    num_classes: int = 7
    in_channels: int = 3
    width_mult: float = 1.0
    depth_mult: float = 1.0
    act: str = "silu"
    stem_reparam: bool = False
    strides: List[int] = (8, 16, 32)


def _round_channels(ch: int, width_mult: float, divisor: int = 8) -> int:
    v = int(ch * width_mult)
    return max(divisor, (v + divisor // 2) // divisor * divisor)


def _round_depth(n: int, depth_mult: float) -> int:
    return max(1, int(round(n * depth_mult)))


class YoloMvX(nn.Module):
    """
    YOLO-MvX: backbone + PAN-FPN + YOLO head (anchor-free).
    Returns raw per-scale predictions suitable for a trainer/decoder.
    """
    def __init__(self, cfg: YoloMvXConfig):
        super().__init__()
        self.cfg = cfg

        # base widths/depths
        base_w = _round_channels(64, cfg.width_mult)
        base_d = _round_depth(2, cfg.depth_mult)

        self.backbone = YoloMvXBackbone(
            in_ch=cfg.in_channels,
            width=base_w,
            depth=base_d,
            act=cfg.act,
            deploy_stem=cfg.stem_reparam,
        )

        c3, c4, c5 = self.backbone.out_channels
        self.neck = PANFPN(c3=c3, c4=c4, c5=c5, width=base_w, depth=max(1, base_d // 2), act=cfg.act)

        head_chs = list(self.neck.out_channels)
        self.head = YoloHead(chs=head_chs, num_classes=cfg.num_classes, act=cfg.act)

        self.strides = list(cfg.strides)

    @staticmethod
    def from_config(d: Dict[str, Any]) -> "YoloMvX":
        cfg = YoloMvXConfig(
            num_classes=int(d.get("num_classes", 7)),
            in_channels=int(d.get("in_channels", 3)),
            width_mult=float(d.get("width_mult", 1.0)),
            depth_mult=float(d.get("depth_mult", 1.0)),
            act=str(d.get("act", "silu")),
            stem_reparam=bool(d.get("stem_reparam", False)),
            strides=list(d.get("strides", [8, 16, 32])),
        )
        return YoloMvX(cfg)

    def forward(self, images: torch.Tensor) -> HeadOut:
        feat = self.backbone(images)
        neck = self.neck(feat.c3, feat.c4, feat.c5)
        out = self.head([neck.p3, neck.p4, neck.p5], strides=self.strides)
        return out

    def switch_to_deploy(self) -> None:
        """
        Fuse stem RepConv blocks into single convs for efficient inference.
        """
        self.backbone.switch_to_deploy()