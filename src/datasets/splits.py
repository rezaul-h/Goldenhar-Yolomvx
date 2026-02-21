from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Set, Tuple

from .io import SplitManifest


def assert_no_overlap(train_list: List[str], val_list: List[str]) -> None:
    tr = set(train_list)
    va = set(val_list)
    overlap = tr.intersection(va)
    if overlap:
        example = next(iter(overlap))
        raise ValueError(f"Split leakage detected: {len(overlap)} overlapping items (e.g., '{example}').")


def validate_split_manifest(manifest: SplitManifest) -> None:
    # Disallow duplicates unless explicitly allowed
    if not manifest.allow_duplicates:
        if len(set(manifest.train_images)) != len(manifest.train_images):
            raise ValueError("Duplicate entries found in train_images but allow_duplicates=false.")
        if len(set(manifest.val_images)) != len(manifest.val_images):
            raise ValueError("Duplicate entries found in val_images but allow_duplicates=false.")

    # Leakage check
    if manifest.enforce_no_leakage:
        assert_no_overlap(manifest.train_images, manifest.val_images)

    # Minimal counts check (warn behavior is up to caller)
    if len(manifest.train_images) == 0 or len(manifest.val_images) == 0:
        # keep as hard error because training/eval would be meaningless
        raise ValueError(
            f"Split manifest has empty lists: n_train={len(manifest.train_images)}, n_val={len(manifest.val_images)}. "
            "Populate train_images and val_images."
        )


def resolve_image_path(root: Path, rel_path: str) -> Path:
    p = root / rel_path
    return p


def resolve_label_path(labels_dir: Path, image_path: Path) -> Path:
    return labels_dir / (image_path.stem + ".txt")