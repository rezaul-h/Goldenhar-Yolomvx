from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from .augment import AugmentConfig, augment_sample, make_rng, read_yolo_txt, write_yolo_txt
from .io import load_dataset_config, load_split_manifest


# -------------------------
# Utilities
# -------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def _copy_or_resize_image(src: Path, dst: Path, img_size: int) -> None:
    """
    Resize to square (img_size x img_size) while *keeping YOLO labels valid* requires also rescaling boxes.
    Here we assume labels are normalized YOLO => no label change needed for resize to square,
    as long as we resize image directly (normalized coords remain consistent).
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(src).convert("RGB")
    img = img.resize((img_size, img_size), resample=Image.BILINEAR)
    img.save(dst, quality=95)


def _copy_label(src_label: Path, dst_label: Path) -> None:
    dst_label.parent.mkdir(parents=True, exist_ok=True)
    if src_label.exists():
        shutil.copyfile(src_label, dst_label)
    else:
        # keep empty label file to preserve pairing
        dst_label.write_text("", encoding="utf-8")


def _label_path_from_image(labels_dir: Path, img_path: Path) -> Path:
    return labels_dir / (img_path.stem + ".txt")


@dataclass(frozen=True)
class BuildPlan:
    dataset_cfg_path: Path
    split_manifest_path: Path
    out_name: str
    out_img_size: int
    n_aug_per_image: int
    global_seed: int
    only_train: bool = True


def build_processed_dataset(plan: BuildPlan) -> Path:
    """
    Build a processed dataset folder:
      data/processed/<out_name>_yolo<sz>/
        images/
        labels/
        aug_images/ (optional)
        aug_labels/ (optional)

    Augmentation is applied strictly on training images (leakage-aware).
    """
    dataset_cfg = load_dataset_config(plan.dataset_cfg_path)
    manifest = load_split_manifest(plan.split_manifest_path)

    root = dataset_cfg.paths.root
    images_dir = root / dataset_cfg.paths.images_dir
    labels_dir = root / dataset_cfg.paths.labels_dir

    out_root = Path("data") / "processed" / f"{plan.out_name}_yolo{plan.out_img_size}"
    out_images = out_root / "images"
    out_labels = out_root / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    # Optional augmentation outputs
    out_aug_images = out_root / "aug_images"
    out_aug_labels = out_root / "aug_labels"
    if plan.n_aug_per_image > 0:
        out_aug_images.mkdir(parents=True, exist_ok=True)
        out_aug_labels.mkdir(parents=True, exist_ok=True)

    # Decide which images to include in processed set
    train_list = list(manifest.train_images)
    val_list = list(manifest.val_images)
    all_list = train_list + ([] if plan.only_train else val_list)

    # Copy/resize base set
    missing_imgs = 0
    for rel in all_list:
        src_img = root / rel
        if not src_img.exists():
            missing_imgs += 1
            continue

        dst_img = out_images / src_img.name
        _copy_or_resize_image(src_img, dst_img, img_size=plan.out_img_size)

        src_lab = _label_path_from_image(labels_dir, src_img)
        dst_lab = out_labels / (src_img.stem + ".txt")
        _copy_label(src_lab, dst_lab)

    if missing_imgs > 0:
        print(f"[WARN] Missing {missing_imgs} images referenced in manifest.")

    # Augment training only
    if plan.n_aug_per_image > 0:
        cfg = AugmentConfig()
        aug_count = 0

        for rel in train_list:
            src_img = root / rel
            if not src_img.exists():
                continue
            src_lab = _label_path_from_image(labels_dir, src_img)

            img = Image.open(src_img).convert("RGB")
            rows = read_yolo_txt(src_lab)

            for k in range(plan.n_aug_per_image):
                # deterministic RNG per image + run + aug index
                rng = make_rng(plan.global_seed + k, image_id=str(rel), run=int(manifest.run))

                img_aug, rows_aug = augment_sample(img, rows, cfg, rng)
                # resize output to target size (labels remain normalized, no change needed)
                img_aug = img_aug.resize((plan.out_img_size, plan.out_img_size), resample=Image.BILINEAR)

                aug_stem = f"{src_img.stem}_aug{k+1:02d}_run{manifest.run}"
                dst_img = out_aug_images / f"{aug_stem}.jpg"
                dst_lab = out_aug_labels / f"{aug_stem}.txt"

                dst_img.parent.mkdir(parents=True, exist_ok=True)
                img_aug.save(dst_img, quality=95)
                write_yolo_txt(dst_lab, rows_aug)
                aug_count += 1

        print(f"[INFO] Augmented samples generated: {aug_count}")

    print(f"[DONE] Processed dataset created at: {out_root.resolve()}")
    return out_root


# -------------------------
# CLI
# -------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build processed YOLO dataset (resize + leakage-aware augmentation).")
    ap.add_argument("--dataset-config", type=str, required=True, help="Path to configs/datasets/D1.yaml or D2.yaml")
    ap.add_argument("--split-manifest", type=str, required=True, help="Path to data/splits/.../run*_seed*.yaml")
    ap.add_argument("--out-name", type=str, required=True, help="Output dataset name prefix (e.g., D1, D2_balanced)")
    ap.add_argument("--img-size", type=int, default=640, help="Square resize size (default: 640)")
    ap.add_argument("--n-aug", type=int, default=0, help="Augmented samples per training image (default: 0)")
    ap.add_argument("--seed", type=int, default=123, help="Global seed for deterministic augmentation")
    ap.add_argument("--only-train", action="store_true", help="If set, only build processed train subset (default: False)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    plan = BuildPlan(
        dataset_cfg_path=Path(args.dataset_config),
        split_manifest_path=Path(args.split_manifest),
        out_name=str(args.out_name),
        out_img_size=int(args.img_size),
        n_aug_per_image=int(args.n_aug),
        global_seed=int(args.seed),
        only_train=bool(args.only_train),
    )
    build_processed_dataset(plan)


if __name__ == "__main__":
    main()