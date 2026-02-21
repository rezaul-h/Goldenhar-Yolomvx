from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_act(name: str) -> nn.Module:
    name = name.lower().strip()
    if name in {"silu", "swish"}:
        return nn.SiLU(inplace=True)
    if name in {"relu"}:
        return nn.ReLU(inplace=True)
    if name in {"gelu"}:
        return nn.GELU()
    raise ValueError(f"Unknown activation: {name}")


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 3,
        s: int = 1,
        p: Optional[int] = None,
        g: int = 1,
        act: str = "silu",
    ):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = get_act(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DWConv(nn.Module):
    """Depthwise separable conv: DW + PW."""
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, act: str = "silu"):
        super().__init__()
        self.dw = ConvBNAct(in_ch, in_ch, k=k, s=s, g=in_ch, act=act)
        self.pw = ConvBNAct(in_ch, out_ch, k=1, s=1, p=0, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class SPPF(nn.Module):
    """SPPF block from YOLOv5: fast spatial pyramid pooling."""
    def __init__(self, in_ch: int, out_ch: int, k: int = 5, act: str = "silu"):
        super().__init__()
        hidden = in_ch // 2
        self.cv1 = ConvBNAct(in_ch, hidden, k=1, s=1, p=0, act=act)
        self.cv2 = ConvBNAct(hidden * 4, out_ch, k=1, s=1, p=0, act=act)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


class CSPBlock(nn.Module):
    """
    Lightweight CSP-style block:
      split -> transform one branch -> concat -> fuse
    """
    def __init__(self, in_ch: int, out_ch: int, n: int = 1, act: str = "silu"):
        super().__init__()
        hidden = out_ch // 2
        self.cv1 = ConvBNAct(in_ch, hidden, k=1, s=1, p=0, act=act)
        self.cv2 = ConvBNAct(in_ch, hidden, k=1, s=1, p=0, act=act)
        self.m = nn.Sequential(*[DWConv(hidden, hidden, k=3, s=1, act=act) for _ in range(n)])
        self.cv3 = ConvBNAct(hidden * 2, out_ch, k=1, s=1, p=0, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv3(torch.cat([y1, y2], dim=1))


class RepConv(nn.Module):
    """
    MobileOne-style re-parameterizable block (training-time multi-branch, inference-time single conv).
    This simplified version supports:
      - 3x3 conv+bn branch
      - 1x1 conv+bn branch
      - identity+bn branch (if in_ch==out_ch and stride==1)

    Call `switch_to_deploy()` after training to fuse to single conv.
    """
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, act: str = "silu", deploy: bool = False):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.k = k
        self.s = s
        self.deploy = deploy
        self.act = get_act(act)

        padding = k // 2

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_ch, out_ch, k, s, padding, bias=True)
        else:
            self.rbr_3x3 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, k, s, padding, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, s, 0, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            self.rbr_identity = None
            if out_ch == in_ch and s == 1:
                self.rbr_identity = nn.BatchNorm2d(in_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            return self.act(self.rbr_reparam(x))

        out = self.rbr_3x3(x) + self.rbr_1x1(x)
        if self.rbr_identity is not None:
            out = out + self.rbr_identity(x)
        return self.act(out)

    @staticmethod
    def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> Tuple[torch.Tensor, torch.Tensor]:
        w = conv.weight
        if conv.bias is None:
            bias = torch.zeros(w.size(0), device=w.device)
        else:
            bias = conv.bias

        gamma = bn.weight
        beta = bn.bias
        mean = bn.running_mean
        var = bn.running_var
        eps = bn.eps

        std = torch.sqrt(var + eps)
        w_fused = w * (gamma / std).reshape(-1, 1, 1, 1)
        b_fused = beta + (bias - mean) * (gamma / std)
        return w_fused, b_fused

    @staticmethod
    def _pad_1x1_to_kxk(w_1x1: torch.Tensor, k: int) -> torch.Tensor:
        if k == 1:
            return w_1x1
        pad = k // 2
        return F.pad(w_1x1, [pad, pad, pad, pad])

    def _fuse_identity_bn(self, bn: nn.BatchNorm2d) -> Tuple[torch.Tensor, torch.Tensor]:
        # identity conv kernel
        in_ch = self.in_ch
        k = self.k
        device = bn.weight.device
        w = torch.zeros((in_ch, in_ch, k, k), device=device)
        mid = k // 2
        for i in range(in_ch):
            w[i, i, mid, mid] = 1.0

        gamma = bn.weight
        beta = bn.bias
        mean = bn.running_mean
        var = bn.running_var
        eps = bn.eps
        std = torch.sqrt(var + eps)

        w_fused = w * (gamma / std).reshape(-1, 1, 1, 1)
        b_fused = beta - mean * (gamma / std)
        return w_fused, b_fused

    def switch_to_deploy(self) -> None:
        if self.deploy:
            return

        w3, b3 = self._fuse_conv_bn(self.rbr_3x3[0], self.rbr_3x3[1])
        w1, b1 = self._fuse_conv_bn(self.rbr_1x1[0], self.rbr_1x1[1])
        w1 = self._pad_1x1_to_kxk(w1, self.k)

        if self.rbr_identity is not None:
            wi, bi = self._fuse_identity_bn(self.rbr_identity)
        else:
            wi = torch.zeros_like(w3)
            bi = torch.zeros_like(b3)

        w = w3 + w1 + wi
        b = b3 + b1 + bi

        padding = self.k // 2
        self.rbr_reparam = nn.Conv2d(self.in_ch, self.out_ch, self.k, self.s, padding, bias=True)
        self.rbr_reparam.weight.data = w
        self.rbr_reparam.bias.data = b

        # drop training branches
        del self.rbr_3x3
        del self.rbr_1x1
        if hasattr(self, "rbr_identity"):
            del self.rbr_identity

        self.deploy = True