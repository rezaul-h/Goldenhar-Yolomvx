from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    def __init__(self, in_ch: int = 3, embed_dim: int = 96, patch: int = 4):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch)

    def forward(self, x):
        x = self.proj(x)  # (B,C,H/patch,W/patch)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B,HW,C)
        return x, (H, W)


class TinyBlock(nn.Module):
    """
    Lightweight transformer block placeholder (not full Swin W-MSA).
    Purpose: stable baseline backbone for your pipeline without extra deps.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        h = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, need_weights=False)
        x = x + h

        h = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + h
        return x


class SwinTinyBackbone(nn.Module):
    """
    Simplified Swin-T style backbone:
    - Patch embedding
    - 4 stages with downsampling
    - lightweight transformer blocks (not shifted windows)
    """
    def __init__(self, embed_dim: int = 96, depths=(2, 2, 6, 2)):
        super().__init__()
        self.patch = PatchEmbed(embed_dim=embed_dim)
        dims = [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8]

        self.stages = nn.ModuleList()
        self.down = nn.ModuleList()

        in_dim = embed_dim
        for si, (d, dim) in enumerate(zip(depths, dims)):
            if si == 0:
                # already embed_dim
                pass
            else:
                # downsample: linear proj + reshape via pooling-like token reduction
                self.down.append(nn.Linear(in_dim, dim))
                in_dim = dim

            blocks = nn.Sequential(*[TinyBlock(in_dim) for _ in range(d)])
            self.stages.append(blocks)

        self.out_dim = dims[-1]

    def forward(self, x):
        x, (H, W) = self.patch(x)  # (B,HW,C)
        # stage 0
        x = self.stages[0](x)

        # stage 1..3 with downsample
        down_i = 0
        for si in range(1, 4):
            # token reduction: take every 4th token as a cheap proxy
            B, N, C = x.shape
            x = x[:, ::4, :]  # crude reduction (fast, deterministic)
            x = self.down[down_i](x)
            down_i += 1
            x = self.stages[si](x)

        return x  # (B,N,out_dim)


class SwinDetectorHead(nn.Module):
    """
    Minimal detection head:
    - global token pooling
    - predict K boxes + class logits (toy baseline)
    This is not a production detector; used for pipeline baselines.
    """
    def __init__(self, in_dim: int, num_classes: int, num_queries: int = 100):
        super().__init__()
        self.num_queries = num_queries
        self.query = nn.Embedding(num_queries, in_dim)
        self.attn = nn.MultiheadAttention(in_dim, num_heads=4, batch_first=True)
        self.cls = nn.Linear(in_dim, num_classes + 1)
        self.box = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, 4),
        )

    def forward(self, tokens):
        B, N, D = tokens.shape
        q = self.query.weight.unsqueeze(0).repeat(B, 1, 1)  # (B,Q,D)
        q, _ = self.attn(q, tokens, tokens, need_weights=False)
        logits = self.cls(q)
        boxes = self.box(q).sigmoid()
        return {"pred_logits": logits, "pred_boxes": boxes}


class SwinTinyDetector(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int = 96, num_queries: int = 100):
        super().__init__()
        self.backbone = SwinTinyBackbone(embed_dim=embed_dim)
        self.head = SwinDetectorHead(self.backbone.out_dim, num_classes=num_classes, num_queries=num_queries)

    def forward(self, images: torch.Tensor):
        tokens = self.backbone(images)
        return self.head(tokens)


def build_swin_tiny(cfg: Dict[str, Any]) -> nn.Module:
    num_classes = int(cfg.get("num_classes", 7))
    embed_dim = int(cfg.get("embed_dim", 96))
    num_queries = int(cfg.get("num_queries", 100))
    return SwinTinyDetector(num_classes=num_classes, embed_dim=embed_dim, num_queries=num_queries)