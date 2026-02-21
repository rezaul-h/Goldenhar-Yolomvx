from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


# -------------------------
# YOLO helpers
# -------------------------
def read_yolo_txt(path: Path) -> List[Tuple[int, float, float, float, float]]:
    """
    YOLO label format: class x_center y_center width height (normalized [0,1]).
    Returns list of tuples.
    """
    if not path.exists():
        return []
    txt = path.read_text(encoding="utf-8").strip()
    if not txt:
        return []
    rows = []
    for line in txt.splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        c = int(float(parts[0]))
        x, y, w, h = map(float, parts[1:])
        rows.append((c, float(x), float(y), float(w), float(h)))
    return rows


def write_yolo_txt(path: Path, rows: List[Tuple[int, float, float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for c, x, y, w, h in rows:
        # enforce numeric stability
        x = float(np.clip(x, 0.0, 1.0))
        y = float(np.clip(y, 0.0, 1.0))
        w = float(np.clip(w, 0.0, 1.0))
        h = float(np.clip(h, 0.0, 1.0))
        lines.append(f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def clamp_valid_boxes(
    rows: List[Tuple[int, float, float, float, float]],
    min_wh: float = 1e-4,
) -> List[Tuple[int, float, float, float, float]]:
    """
    Removes degenerate boxes after augmentation and clamps to [0,1].
    """
    out = []
    for c, x, y, w, h in rows:
        w = float(np.clip(w, 0.0, 1.0))
        h = float(np.clip(h, 0.0, 1.0))
        x = float(np.clip(x, 0.0, 1.0))
        y = float(np.clip(y, 0.0, 1.0))
        if w < min_wh or h < min_wh:
            continue
        out.append((int(c), x, y, w, h))
    return out


# -------------------------
# Augmentation config
# -------------------------
@dataclass(frozen=True)
class AugmentConfig:
    """
    Conservative augmentation for craniofacial imagery (safe defaults):
    - horizontal flip (p)
    - small rotation (degrees)
    - translation (fraction of width/height)
    - mild brightness/contrast jitter
    - gaussian noise (low)
    """
    p_hflip: float = 0.5
    rot_deg: float = 7.0
    translate: float = 0.04
    brightness: float = 0.08
    contrast: float = 0.08
    noise_std: float = 3.0   # in 0..255 domain
    keep_prob: float = 1.0   # allow skipping augmentation for some samples if desired


# -------------------------
# Geometry transforms for YOLO boxes
# -------------------------
def _yolo_to_xyxy(rows: List[Tuple[int, float, float, float, float]], w: int, h: int):
    xyxy = []
    cls = []
    for c, x, y, bw, bh in rows:
        x1 = (x - bw / 2.0) * w
        y1 = (y - bh / 2.0) * h
        x2 = (x + bw / 2.0) * w
        y2 = (y + bh / 2.0) * h
        xyxy.append([x1, y1, x2, y2])
        cls.append(int(c))
    if len(xyxy) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.asarray(xyxy, dtype=np.float32), np.asarray(cls, dtype=np.int64)


def _xyxy_to_yolo(xyxy: np.ndarray, cls: np.ndarray, w: int, h: int):
    rows = []
    for i in range(xyxy.shape[0]):
        x1, y1, x2, y2 = xyxy[i]
        x1 = float(np.clip(x1, 0.0, w - 1.0))
        x2 = float(np.clip(x2, 0.0, w - 1.0))
        y1 = float(np.clip(y1, 0.0, h - 1.0))
        y2 = float(np.clip(y2, 0.0, h - 1.0))
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        if bw <= 1.0 or bh <= 1.0:
            continue
        xc = (x1 + x2) / 2.0 / w
        yc = (y1 + y2) / 2.0 / h
        bw_n = bw / w
        bh_n = bh / h
        rows.append((int(cls[i]), float(xc), float(yc), float(bw_n), float(bh_n)))
    return rows


def _affine_matrix(rot_deg: float, tx: float, ty: float, cx: float, cy: float):
    """
    2x3 affine matrix for rotation around center plus translation.
    """
    theta = np.deg2rad(rot_deg)
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))

    # rotation about center: translate(-c) -> rotate -> translate(c) then add tx,ty
    a = cos_t
    b = -sin_t
    d = sin_t
    e = cos_t

    c = cx - a * cx - b * cy + tx
    f = cy - d * cx - e * cy + ty
    return np.array([[a, b, c], [d, e, f]], dtype=np.float32)


def _apply_affine_to_boxes(xyxy: np.ndarray, M: np.ndarray):
    """
    Apply affine transform to each box by transforming its four corners
    and taking min/max.
    """
    if xyxy.shape[0] == 0:
        return xyxy

    out = np.zeros_like(xyxy, dtype=np.float32)
    for i in range(xyxy.shape[0]):
        x1, y1, x2, y2 = xyxy[i]
        corners = np.array(
            [[x1, y1, 1.0], [x2, y1, 1.0], [x2, y2, 1.0], [x1, y2, 1.0]],
            dtype=np.float32,
        )  # (4,3)
        tc = (M @ corners.T).T  # (4,2)
        xs = tc[:, 0]
        ys = tc[:, 1]
        out[i, 0] = float(xs.min())
        out[i, 1] = float(ys.min())
        out[i, 2] = float(xs.max())
        out[i, 3] = float(ys.max())
    return out


# -------------------------
# Pixel transforms
# -------------------------
def _jitter(img_arr: np.ndarray, rng: np.random.Generator, brightness: float, contrast: float) -> np.ndarray:
    # img_arr uint8
    alpha = float(rng.uniform(1.0 - contrast, 1.0 + contrast))  # contrast
    beta = float(rng.uniform(-brightness, brightness)) * 255.0  # brightness shift
    out = np.clip(alpha * img_arr.astype(np.float32) + beta, 0.0, 255.0).astype(np.uint8)
    return out


def _add_noise(img_arr: np.ndarray, rng: np.random.Generator, std: float) -> np.ndarray:
    if std <= 0:
        return img_arr
    noise = rng.normal(0.0, std, size=img_arr.shape).astype(np.float32)
    out = np.clip(img_arr.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)
    return out


# -------------------------
# Main augmentation API
# -------------------------
def augment_sample(
    img: Image.Image,
    yolo_rows: List[Tuple[int, float, float, float, float]],
    cfg: AugmentConfig,
    rng: np.random.Generator,
) -> Tuple[Image.Image, List[Tuple[int, float, float, float, float]]]:
    """
    Apply conservative augmentation and return new (image, yolo_rows).
    Deterministic given rng state.
    """
    if rng.random() > cfg.keep_prob:
        return img, yolo_rows

    w, h = img.size
    img_arr = np.asarray(img, dtype=np.uint8)

    # Convert boxes to xyxy in pixels
    xyxy, cls = _yolo_to_xyxy(yolo_rows, w=w, h=h)

    # HFlip
    do_hflip = (rng.random() < cfg.p_hflip)
    if do_hflip:
        img_arr = img_arr[:, ::-1, :].copy()
        if xyxy.shape[0] > 0:
            x1 = xyxy[:, 0].copy()
            x2 = xyxy[:, 2].copy()
            xyxy[:, 0] = (w - 1.0) - x2
            xyxy[:, 2] = (w - 1.0) - x1

    # Rotation + translation (single affine)
    rot = float(rng.uniform(-cfg.rot_deg, cfg.rot_deg))
    tx = float(rng.uniform(-cfg.translate, cfg.translate)) * w
    ty = float(rng.uniform(-cfg.translate, cfg.translate)) * h
    cx, cy = (w - 1.0) / 2.0, (h - 1.0) / 2.0
    M = _affine_matrix(rot_deg=rot, tx=tx, ty=ty, cx=cx, cy=cy)

    # Apply affine to image via PIL (dependency-free)
    img_aff = Image.fromarray(img_arr)
    img_aff = img_aff.transform(
        size=(w, h),
        method=Image.AFFINE,
        data=(float(M[0, 0]), float(M[0, 1]), float(M[0, 2]), float(M[1, 0]), float(M[1, 1]), float(M[1, 2])),
        resample=Image.BILINEAR,
        fillcolor=(0, 0, 0),
    )
    img_arr = np.asarray(img_aff, dtype=np.uint8)

    # Apply affine to boxes
    if xyxy.shape[0] > 0:
        xyxy = _apply_affine_to_boxes(xyxy, M)

    # Photometric
    img_arr = _jitter(img_arr, rng, brightness=cfg.brightness, contrast=cfg.contrast)
    img_arr = _add_noise(img_arr, rng, std=cfg.noise_std)

    # Convert boxes back to YOLO normalized
    yolo_new = _xyxy_to_yolo(xyxy, cls, w=w, h=h)
    yolo_new = clamp_valid_boxes(yolo_new)

    return Image.fromarray(img_arr), yolo_new


def make_rng(global_seed: int, image_id: str, run: int) -> np.random.Generator:
    """
    Deterministic per-image RNG:
      seed = hash(global_seed, run, image_id)
    """
    h = (hash((int(global_seed), int(run), str(image_id))) % (2**32 - 1))
    return np.random.default_rng(h)