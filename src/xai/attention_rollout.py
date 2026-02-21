from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .heatmaps import normalize_map, resize_map


@dataclass
class RolloutOutput:
    heatmap: torch.Tensor  # (H,W) in [0,1]


class AttentionRollout:
    """
    Attention rollout for transformer-style models.

    Requirements:
      - You supply a list of attention matrices (per layer) of shape (B, heads, tokens, tokens)
      - Or you provide module hooks that can capture them externally.

    This class implements the rollout computation and returns a spatial map for image tokens.
    """

    def __init__(self, discard_ratio: float = 0.0, head_fusion: str = "mean"):
        self.discard_ratio = float(discard_ratio)
        self.head_fusion = head_fusion.lower().strip()

    def _fuse_heads(self, attn: torch.Tensor) -> torch.Tensor:
        # attn: (heads, T, T)
        if self.head_fusion == "mean":
            return attn.mean(dim=0)
        if self.head_fusion == "max":
            return attn.max(dim=0).values
        if self.head_fusion == "min":
            return attn.min(dim=0).values
        raise ValueError("head_fusion must be one of: mean, max, min")

    def __call__(
        self,
        attn_layers: List[torch.Tensor],
        token_hw: Tuple[int, int],
        out_hw: Tuple[int, int],
        cls_token: bool = True,
    ) -> RolloutOutput:
        """
        attn_layers: list of (1, heads, T, T)
        token_hw: (h_tokens, w_tokens) for image tokens (excluding cls if present)
        out_hw: (H,W) output heatmap size
        cls_token: whether token 0 is CLS token (common in ViT)
        """
        if len(attn_layers) == 0:
            raise ValueError("attn_layers cannot be empty.")

        # Build joint attention
        joint = None
        for attn in attn_layers:
            if attn.dim() != 4 or attn.shape[0] != 1:
                raise ValueError("Each attention tensor must be shape (1, heads, T, T).")

            a = attn[0]  # (heads,T,T)
            a = self._fuse_heads(a)  # (T,T)

            # discard low attentions (optional)
            if self.discard_ratio > 0:
                flat = a.flatten()
                k = int(flat.numel() * self.discard_ratio)
                if k > 0:
                    thresh = torch.kthvalue(flat, k).values
                    a = torch.where(a <= thresh, torch.zeros_like(a), a)

            # add residual and normalize
            T = a.shape[0]
            a = a + torch.eye(T, device=a.device)
            a = a / (a.sum(dim=-1, keepdim=True) + 1e-8)

            joint = a if joint is None else a @ joint

        # Extract attention from CLS to image tokens or average token-to-token
        T = joint.shape[0]
        if cls_token:
            # tokens 1.. are image tokens
            v = joint[0, 1:]  # (T-1,)
        else:
            v = joint.mean(dim=0)  # (T,)

        h_t, w_t = token_hw
        if cls_token and v.numel() != h_t * w_t:
            raise ValueError("token_hw does not match number of image tokens.")
        if not cls_token and v.numel() < h_t * w_t:
            raise ValueError("token_hw inconsistent with tokens.")

        v = v[: h_t * w_t].reshape(h_t, w_t)
        v = normalize_map(v)
        v = resize_map(v, out_hw)
        v = normalize_map(v)
        return RolloutOutput(heatmap=v)