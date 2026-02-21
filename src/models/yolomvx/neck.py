from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBNAct, CSPBlock, DWConv


@dataclass(frozen=True)
class NeckOut:
    p3: torch.Tensor  # stride 8
    p4: torch.Tensor  # stride 16
    p5: torch.Tensor  # stride 32


class PANFPN(nn.Module):
    """
    Standard PAN-FPN neck for 3 scales:
      top-down FPN then bottom-up PAN
    """
    def __init__(self, c3: int, c4: int, c5: int, width: int, depth: int = 1, act: str = "silu"):
        super().__init__()
        # lateral projections
        self.lat5 = ConvBNAct(c5, width * 4, k=1, s=1, p=0, act=act)
        self.lat4 = ConvBNAct(c4, width * 4, k=1, s=1, p=0, act=act)
        self.lat3 = ConvBNAct(c3, width * 2, k=1, s=1, p=0, act=act)

        # top-down
        self.fpn4 = CSPBlock(width * 8, width * 4, n=depth, act=act)
        self.fpn3 = CSPBlock(width * 6, width * 2, n=depth, act=act)

        # bottom-up
        self.down3 = DWConv(width * 2, width * 2, k=3, s=2, act=act)
        self.pan4 = CSPBlock(width * 6, width * 4, n=depth, act=act)

        self.down4 = DWConv(width * 4, width * 4, k=3, s=2, act=act)
        self.pan5 = CSPBlock(width * 8, width * 4, n=depth, act=act)

        self.out_channels = [width * 2, width * 4, width * 4]

    def forward(self, c3: torch.Tensor, c4: torch.Tensor, c5: torch.Tensor) -> NeckOut:
        p5 = self.lat5(c5)
        p4 = self.lat4(c4)
        p3 = self.lat3(c3)

        # FPN: p5 -> p4
        p5_up = F.interpolate(p5, size=p4.shape[-2:], mode="nearest")
        p4 = self.fpn4(torch.cat([p4, p5_up], dim=1))

        # FPN: p4 -> p3
        p4_up = F.interpolate(p4, size=p3.shape[-2:], mode="nearest")
        p3 = self.fpn3(torch.cat([p3, p4_up], dim=1))

        # PAN: p3 -> p4
        p3_down = self.down3(p3)
        p4 = self.pan4(torch.cat([p3_down, p4], dim=1))

        # PAN: p4 -> p5
        p4_down = self.down4(p4)
        p5 = self.pan5(torch.cat([p4_down, p5], dim=1))

        return NeckOut(p3=p3, p4=p4, p5=p5)