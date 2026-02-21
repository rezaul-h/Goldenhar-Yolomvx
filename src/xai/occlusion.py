from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F

from .heatmaps import normalize_map, resize_map


@dataclass
class OcclusionOutput:
    heatmap: torch.Tensor  # (H,W) in [0,1]
    baseline_score: float


class OcclusionSensitivity:
    """
    Model-agnostic occlusion sensitivity.

    You provide a `score_fn(model, x)` that outputs a scalar tensor to explain.
    This works for detectors too if score_fn aggregates detection confidence
    for a target class/box.
    """

    def __init__(self, patch: int = 32, stride: int = 16, fill: float = 0.0):
        self.patch = int(patch)
        self.stride = int(stride)
        self.fill = float(fill)

    @torch.no_grad()
    def __call__(
        self,
        model,
        x: torch.Tensor,
        score_fn: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor],
    ) -> OcclusionOutput:
        """
        x: (1,3,H,W) in [0,1]
        """
        model.eval()
        H, W = x.shape[2], x.shape[3]
        base = score_fn(model, x).detach()
        baseline_score = float(base.cpu().item())

        ph, pw = self.patch, self.patch
        sh, sw = self.stride, self.stride

        out_h = max(1, (H - ph) // sh + 1)
        out_w = max(1, (W - pw) // sw + 1)
        heat = torch.zeros((out_h, out_w), device=x.device)

        for i in range(out_h):
            for j in range(out_w):
                y1 = i * sh
                x1 = j * sw
                y2 = min(y1 + ph, H)
                x2 = min(x1 + pw, W)

                xo = x.clone()
                xo[:, :, y1:y2, x1:x2] = self.fill
                s = score_fn(model, xo).detach()
                # drop in score indicates importance (higher drop => more important)
                heat[i, j] = (base - s).clamp(min=0)

        heat = normalize_map(heat)
        heat = resize_map(heat, (H, W))
        heat = normalize_map(heat)
        return OcclusionOutput(heatmap=heat, baseline_score=baseline_score)