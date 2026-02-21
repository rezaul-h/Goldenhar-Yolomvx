from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def to_numpy_img(img: torch.Tensor) -> np.ndarray:
    """
    img: (3,H,W) float tensor in [0,1] (recommended).
    returns: uint8 HxWx3
    """
    x = img.detach().cpu().clamp(0, 1).numpy()
    x = (x.transpose(1, 2, 0) * 255.0).astype(np.uint8)
    return x


def normalize_map(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize a heatmap to [0,1] per sample.
    x: (H,W) or (B,1,H,W)
    """
    if x.dim() == 2:
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + eps)
    if x.dim() == 4:
        mn = x.amin(dim=(2, 3), keepdim=True)
        mx = x.amax(dim=(2, 3), keepdim=True)
        return (x - mn) / (mx - mn + eps)
    raise ValueError("normalize_map expects 2D or 4D tensor.")


def resize_map(hm: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
    """
    hm: (H,W) or (1,1,H,W)
    """
    if hm.dim() == 2:
        hm = hm[None, None, ...]
    out = F.interpolate(hm, size=size_hw, mode="bilinear", align_corners=False)
    return out[0, 0]


def _viridis_like(heat: np.ndarray) -> np.ndarray:
    """
    Lightweight viridis-like colormap without matplotlib dependency.
    Input: heat in [0,1], shape (H,W)
    Output: RGB uint8 (H,W,3)
    """
    h = np.clip(heat, 0.0, 1.0)
    # Piecewise polynomial-ish approximation (simple, stable)
    r = np.clip(1.5*h - 0.2, 0, 1)
    g = np.clip(1.8*np.sqrt(h) - 0.1, 0, 1)
    b = np.clip(1.2*(1 - h) + 0.05, 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def overlay_heatmap(
    image_rgb_u8: np.ndarray,
    heat_01: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Q1-friendly overlay: keeps natural image + semi-transparent heatmap.
    """
    hm_rgb = _viridis_like(heat_01)
    img = image_rgb_u8.astype(np.float32)
    hm = hm_rgb.astype(np.float32)
    out = (1 - alpha) * img + alpha * hm
    return np.clip(out, 0, 255).astype(np.uint8)


def save_image(arr_u8: np.ndarray, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr_u8).save(path)
    return path