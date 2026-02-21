from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .constants import NUM_CLASSES
from .io import DatasetConfig, SplitManifest
from .splits import resolve_image_path, resolve_label_path, validate_split_manifest


@dataclass(frozen=True)
class DetectionSample:
    image: torch.Tensor                # (C,H,W), float32, [0,1]
    boxes_xyxy: torch.Tensor           # (N,4), float32, pixel coords
    labels: torch.Tensor               # (N,), int64
    image_path: str


def _read_yolo_txt(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """
    YOLO format per line: class x_center y_center width height (all normalized in [0,1])
    """
    if not label_path.exists():
        return []
    txt = label_path.read_text(encoding="utf-8").strip()
    if not txt:
        return []
    rows = []
    for line in txt.splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        c = int(float(parts[0]))
        x, y, w, h = map(float, parts[1:])
        rows.append((c, x, y, w, h))
    return rows


def _yolo_to_xyxy(rows: List[Tuple[int, float, float, float, float]], img_w: int, img_h: int) -> Tuple[np.ndarray, np.ndarray]:
    boxes = []
    labels = []
    for c, x, y, w, h in rows:
        if c < 0 or c >= NUM_CLASSES:
            continue
        # clamp to [0,1] to avoid exploding boxes due to small annotation noise
        x = float(np.clip(x, 0.0, 1.0))
        y = float(np.clip(y, 0.0, 1.0))
        w = float(np.clip(w, 0.0, 1.0))
        h = float(np.clip(h, 0.0, 1.0))

        x1 = (x - w / 2.0) * img_w
        y1 = (y - h / 2.0) * img_h
        x2 = (x + w / 2.0) * img_w
        y2 = (y + h / 2.0) * img_h

        # ensure valid ordering
        x1, x2 = (min(x1, x2), max(x1, x2))
        y1, y2 = (min(y1, y2), max(y1, y2))

        # clamp to image bounds
        x1 = float(np.clip(x1, 0.0, img_w - 1.0))
        x2 = float(np.clip(x2, 0.0, img_w - 1.0))
        y1 = float(np.clip(y1, 0.0, img_h - 1.0))
        y2 = float(np.clip(y2, 0.0, img_h - 1.0))

        # drop degenerate boxes
        if (x2 - x1) < 1.0 or (y2 - y1) < 1.0:
            continue

        boxes.append([x1, y1, x2, y2])
        labels.append(c)

    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    return np.asarray(boxes, dtype=np.float32), np.asarray(labels, dtype=np.int64)


class YoloDetectionDataset(Dataset):
    """
    Minimal, reproducible dataset for YOLO-format labels using split manifests.

    - Loads images from manifest.train_images / manifest.val_images (relative paths).
    - Loads labels from <labels_dir>/<image_stem>.txt
    - Converts YOLO (cx,cy,w,h) to pixel xyxy boxes.

    `transform` is a callable that takes (PIL.Image, boxes_xyxy, labels) and returns transformed tensors.
    """

    def __init__(
        self,
        dataset_cfg: DatasetConfig,
        split_manifest: SplitManifest,
        mode: str,  # "train" | "val"
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        if mode not in {"train", "val"}:
            raise ValueError("mode must be 'train' or 'val'.")

        validate_split_manifest(split_manifest)

        self.dataset_cfg = dataset_cfg
        self.manifest = split_manifest
        self.mode = mode
        self.transform = transform

        self.root = dataset_cfg.paths.root
        self.images_dir = self.root / dataset_cfg.paths.images_dir
        self.labels_dir = self.root / dataset_cfg.paths.labels_dir

        rel_list = split_manifest.train_images if mode == "train" else split_manifest.val_images
        self.image_relpaths = list(rel_list)

        # quick existence scan (fail fast)
        missing = []
        for rel in self.image_relpaths[:200]:  # cap initial scan for speed
            p = resolve_image_path(self.root, rel)
            if not p.exists():
                missing.append(rel)
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} image files (showing up to 3): {missing[:3]}")

    def __len__(self) -> int:
        return len(self.image_relpaths)

    def __getitem__(self, idx: int) -> DetectionSample:
        rel = self.image_relpaths[idx]
        img_path = resolve_image_path(self.root, rel)
        if not img_path.exists():
            raise FileNotFoundError(str(img_path))

        # PIL load (RGB)
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        label_path = resolve_label_path(self.labels_dir, img_path)
        rows = _read_yolo_txt(label_path)
        boxes, labels = _yolo_to_xyxy(rows, img_w=w, img_h=h)

        if self.transform is not None:
            img_t, boxes_t, labels_t = self.transform(img, boxes, labels)
        else:
            img_arr = np.asarray(img, dtype=np.float32) / 255.0
            img_t = torch.from_numpy(img_arr).permute(2, 0, 1).contiguous()
            boxes_t = torch.from_numpy(boxes)
            labels_t = torch.from_numpy(labels)

        return DetectionSample(
            image=img_t,
            boxes_xyxy=boxes_t,
            labels=labels_t.to(torch.int64),
            image_path=str(img_path),
        )


def detection_collate_fn(batch: List[DetectionSample]) -> Dict[str, torch.Tensor | List[torch.Tensor] | List[str]]:
    """
    Collate for variable-size detection targets.
    Returns:
      images: (B,C,H,W)
      targets: list[dict(boxes, labels)]
      paths: list[str]
    """
    images = torch.stack([b.image for b in batch], dim=0)
    targets = [{"boxes": b.boxes_xyxy, "labels": b.labels} for b in batch]
    paths = [b.image_path for b in batch]
    return {"images": images, "targets": targets, "paths": paths}