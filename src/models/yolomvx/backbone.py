from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn

from .blocks import CSPBlock, DWConv, RepConv, SPPF


@dataclass(frozen=True)
class BackboneOut:
    c3: torch.Tensor  # stride 8
    c4: torch.Tensor  # stride 16
    c5: torch.Tensor  # stride 32


class MobileOneStem(nn.Module):
    """
    Efficient stem with RepConv blocks (re-parameterizable).
    """
    def __init__(self, in_ch: int, out_ch: int, act: str = "silu", deploy: bool = False):
        super().__init__()
        mid = out_ch // 2
        self.block1 = RepConv(in_ch, mid, k=3, s=2, act=act, deploy=deploy)   # /2
        self.block2 = RepConv(mid, out_ch, k=3, s=2, act=act, deploy=deploy)  # /4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        return x

    def switch_to_deploy(self) -> None:
        self.block1.switch_to_deploy()
        self.block2.switch_to_deploy()


class YoloMvXBackbone(nn.Module):
    """
    Hierarchical backbone:
      stem -> stage3 (/8) -> stage4 (/16) -> stage5 (/32) + SPPF
    """
    def __init__(
        self,
        in_ch: int = 3,
        width: int = 64,
        depth: int = 2,
        act: str = "silu",
        deploy_stem: bool = False,
    ):
        super().__init__()
        self.stem = MobileOneStem(in_ch, width, act=act, deploy=deploy_stem)

        # /8
        self.down3 = DWConv(width, width * 2, k=3, s=2, act=act)
        self.stage3 = CSPBlock(width * 2, width * 2, n=depth, act=act)

        # /16
        self.down4 = DWConv(width * 2, width * 4, k=3, s=2, act=act)
        self.stage4 = CSPBlock(width * 4, width * 4, n=max(1, depth + 1), act=act)

        # /32
        self.down5 = DWConv(width * 4, width * 8, k=3, s=2, act=act)
        self.stage5 = CSPBlock(width * 8, width * 8, n=max(1, depth + 1), act=act)
        self.sppf = SPPF(width * 8, width * 8, k=5, act=act)

        self.out_channels = [width * 2, width * 4, width * 8]

    def forward(self, x: torch.Tensor) -> BackboneOut:
        x = self.stem(x)              # /4
        c3 = self.stage3(self.down3(x))  # /8
        c4 = self.stage4(self.down4(c3)) # /16
        c5 = self.sppf(self.stage5(self.down5(c4)))  # /32
        return BackboneOut(c3=c3, c4=c4, c5=c5)

    def switch_to_deploy(self) -> None:
        self.stem.switch_to_deploy()