from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


def _require(d: Dict[str, Any], key: str) -> Any:
    if key not in d:
        raise KeyError(f"Missing required key: '{key}'")
    return d[key]


@dataclass(frozen=True)
class DatasetPaths:
    root: Path
    images_dir: str
    labels_dir: str
    path_style: str = "relative_to_root"  # consistent with split manifests


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    task: str
    img_size: int
    class_names: List[str]
    num_classes: int
    paths: DatasetPaths


@dataclass(frozen=True)
class SplitManifest:
    dataset: str
    split_name: str
    run: int
    seed: int
    train_ratio: float
    val_ratio: float
    paths: DatasetPaths
    enforce_no_leakage: bool
    augmentation_train_only: bool
    allow_duplicates: bool
    train_images: List[str]
    val_images: List[str]


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataset_config(path: str | Path) -> DatasetConfig:
    path = Path(path)
    d = load_yaml(path)

    dataset = _require(d, "dataset")
    paths = _require(d, "paths")
    classes = _require(d, "classes")

    cfg = DatasetConfig(
        name=str(_require(dataset, "name")),
        task=str(_require(dataset, "task")),
        img_size=int(_require(dataset, "img_size")),
        class_names=list(_require(classes, "names")),
        num_classes=int(_require(classes, "num_classes")),
        paths=DatasetPaths(
            root=Path(_require(paths, "root")),
            images_dir=str(_require(paths, "images_dir")),
            labels_dir=str(_require(paths, "labels_dir")),
            path_style=str(paths.get("path_style", "relative_to_root")),
        ),
    )
    if cfg.task not in {"detection"}:
        raise ValueError(f"Unsupported task='{cfg.task}'. Expected 'detection'.")
    if cfg.num_classes != len(cfg.class_names):
        raise ValueError("num_classes != len(class_names)")
    return cfg


def load_split_manifest(path: str | Path) -> SplitManifest:
    path = Path(path)
    d = load_yaml(path)

    ratio = _require(d, "split_ratio")
    paths = _require(d, "paths")
    integrity = _require(d, "integrity")

    manifest = SplitManifest(
        dataset=str(_require(d, "dataset")),
        split_name=str(_require(d, "split_name")),
        run=int(_require(d, "run")),
        seed=int(_require(d, "seed")),
        train_ratio=float(_require(ratio, "train")),
        val_ratio=float(_require(ratio, "val")),
        paths=DatasetPaths(
            root=Path(_require(paths, "root")),
            images_dir=str(_require(paths, "images_dir")),
            labels_dir=str(_require(paths, "labels_dir")),
            path_style=str(_require(paths, "path_style")),
        ),
        enforce_no_leakage=bool(_require(integrity, "enforce_no_leakage")),
        augmentation_train_only=bool(_require(integrity, "augmentation_train_only")),
        allow_duplicates=bool(_require(integrity, "allow_duplicates")),
        train_images=list(d.get("train_images", [])),
        val_images=list(d.get("val_images", [])),
    )

    # basic validation
    if manifest.train_ratio + manifest.val_ratio < 0.999 or manifest.train_ratio + manifest.val_ratio > 1.001:
        raise ValueError("train_ratio + val_ratio must be ~1.0")
    if manifest.paths.path_style != "relative_to_root":
        raise ValueError("Only 'relative_to_root' is supported for manifests in this repo.")
    if manifest.run < 1:
        raise ValueError("run must be >= 1")
    return manifest