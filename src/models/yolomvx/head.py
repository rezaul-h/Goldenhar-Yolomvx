from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn

from .blocks import ConvBNAct, DWConv


@dataclass(frozen=True)
class HeadOut:
    # per-scale raw preds in shape (B, A, H, W, K)
    # we use A=1 (anchor-free) and K = 4(box) + 1(obj) + C(cls)
    preds: List[torch.Tensor]
    strides: List[int]


class YoloHead(nn.Module):
    """
    Anchor-free YOLO-style head:
      for each scale: predict [tx,ty,tw,th,obj,cls...]
    """
    def __init__(self, chs: List[int], num_classes: int, act: str = "silu"):
        super().__init__()
        self.num_classes = num_classes
        self.out_dim = 4 + 1 + num_classes

        self.stems = nn.ModuleList([ConvBNAct(c, c, k=1, s=1, p=0, act=act) for c in chs])
        self.cls_convs = nn.ModuleList([DWConv(c, c, k=3, s=1, act=act) for c in chs])
        self.reg_convs = nn.ModuleList([DWConv(c, c, k=3, s=1, act=act) for c in chs])

        self.cls_preds = nn.ModuleList([nn.Conv2d(c, num_classes, kernel_size=1) for c in chs])
        self.obj_preds = nn.ModuleList([nn.Conv2d(c, 1, kernel_size=1) for c in chs])
        self.box_preds = nn.ModuleList([nn.Conv2d(c, 4, kernel_size=1) for c in chs])

    def forward(self, feats: List[torch.Tensor], strides: List[int]) -> HeadOut:
        preds = []
        for i, x in enumerate(feats):
            x = self.stems[i](x)
            cls_feat = self.cls_convs[i](x)
            reg_feat = self.reg_convs[i](x)

            cls = self.cls_preds[i](cls_feat)
            obj = self.obj_preds[i](reg_feat)
            box = self.box_preds[i](reg_feat)

            # (B, K, H, W) -> (B, 1, H, W, K)
            p = torch.cat([box, obj, cls], dim=1)
            p = p.permute(0, 2, 3, 1).contiguous().unsqueeze(1)
            preds.append(p)
        return HeadOut(preds=preds, strides=strides)