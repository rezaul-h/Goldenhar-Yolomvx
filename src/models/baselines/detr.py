from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, depth: int = 3):
        super().__init__()
        layers = []
        d = in_dim
        for i in range(depth - 1):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DETRTiny(nn.Module):
    """
    Minimal DETR-like model for baseline comparisons.
    - Backbone: small conv stem
    - Transformer encoder-decoder
    - Fixed number of object queries
    - Heads: class logits + box (cx,cy,w,h) in [0,1]
    This is NOT a full DETR reproduction; it is a light reference baseline.
    """

    def __init__(self, num_classes: int, hidden_dim: int = 256, num_queries: int = 100):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

        # Tiny backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, hidden_dim, 1),
        )

        self.pos_embed = nn.Parameter(torch.randn(1, hidden_dim, 50, 50) * 0.02)  # interpolated

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 4)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 4)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)

        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.class_head = nn.Linear(hidden_dim, num_classes + 1)  # +1 for "no object"
        self.box_head = MLP(hidden_dim, hidden_dim, 4, depth=3)

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        images: (B,3,H,W)
        returns:
          pred_logits: (B,Q,C+1)
          pred_boxes:  (B,Q,4) in [0,1] cxcywh
        """
        B = images.size(0)
        feat = self.backbone(images)  # (B,hidden,H',W')

        # positional embedding interpolation
        pe = F.interpolate(self.pos_embed, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        feat = feat + pe

        H, W = feat.shape[-2:]
        src = feat.flatten(2).permute(2, 0, 1)  # (HW,B,D)

        memory = self.encoder(src)  # (HW,B,D)

        query = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)  # (Q,B,D)
        tgt = torch.zeros_like(query)

        hs = self.decoder(tgt, memory)  # (Q,B,D)
        hs = hs.permute(1, 0, 2)        # (B,Q,D)

        logits = self.class_head(hs)
        boxes = self.box_head(hs).sigmoid()

        return {"pred_logits": logits, "pred_boxes": boxes}


def build_detr_tiny(cfg: Dict[str, Any]) -> nn.Module:
    num_classes = int(cfg.get("num_classes", 7))
    hidden_dim = int(cfg.get("hidden_dim", 256))
    num_queries = int(cfg.get("num_queries", 100))
    return DETRTiny(num_classes=num_classes, hidden_dim=hidden_dim, num_queries=num_queries)