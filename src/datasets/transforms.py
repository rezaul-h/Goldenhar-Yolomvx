from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
import torch
from PIL import Image


def _to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def build_eval_transforms(img_size: int) -> Callable:
    """
    Deterministic transforms: resize to square (img_size x img_size).
    Boxes are scaled accordingly.
    """

    def _tf(img: Image.Image, boxes_xyxy: np.ndarray, labels: np.ndarray):
        w0, h0 = img.size
        img_r = img.resize((img_size, img_size), resample=Image.BILINEAR)
        sx = img_size / float(w0)
        sy = img_size / float(h0)

        boxes = boxes_xyxy.copy()
        if boxes.shape[0] > 0:
            boxes[:, [0, 2]] *= sx
            boxes[:, [1, 3]] *= sy

        return _to_tensor(img_r), torch.from_numpy(boxes.astype(np.float32)), torch.from_numpy(labels.astype(np.int64))

    return _tf


def build_train_transforms(img_size: int) -> Callable:
    """
    Minimal training transforms (safe defaults for medical/facial imagery):
    - resize to img_size
    - horizontal flip with p=0.5
    - mild brightness/contrast jitter
    No mosaic/mixup here (should be handled in the trainer/augmentor if needed).
    """

    rng = np.random.default_rng()

    def _tf(img: Image.Image, boxes_xyxy: np.ndarray, labels: np.ndarray):
        w0, h0 = img.size

        # resize
        img_r = img.resize((img_size, img_size), resample=Image.BILINEAR)
        sx = img_size / float(w0)
        sy = img_size / float(h0)

        boxes = boxes_xyxy.copy()
        if boxes.shape[0] > 0:
            boxes[:, [0, 2]] *= sx
            boxes[:, [1, 3]] *= sy

        # horizontal flip
        if rng.random() < 0.5:
            img_r = img_r.transpose(Image.FLIP_LEFT_RIGHT)
            if boxes.shape[0] > 0:
                x1 = boxes[:, 0].copy()
                x2 = boxes[:, 2].copy()
                boxes[:, 0] = img_size - 1.0 - x2
                boxes[:, 2] = img_size - 1.0 - x1

        # light color jitter (brightness/contrast)
        # implemented as affine on pixel values to stay dependency-free
        img_arr = np.asarray(img_r, dtype=np.float32)
        alpha = float(rng.uniform(0.9, 1.1))  # contrast
        beta = float(rng.uniform(-10.0, 10.0))  # brightness shift in [0..255] domain
        img_arr = np.clip(alpha * img_arr + beta, 0.0, 255.0).astype(np.uint8)
        img_r = Image.fromarray(img_arr)

        return _to_tensor(img_r), torch.from_numpy(boxes.astype(np.float32)), torch.from_numpy(labels.astype(np.int64))

    return _tf