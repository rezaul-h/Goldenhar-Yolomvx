from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..datasets.io import load_dataset_config, load_split_manifest
from ..datasets.yolo_detection import YoloDetectionDataset, detection_collate_fn
from ..datasets.transforms import build_train_transforms, build_eval_transforms

from ..models.yolomvx import YoloMvX, YoloMvXLoss
from ..models.baselines import build_baseline
from ..models.baselines.base import BaselineAdapter

from .optim import build_optimizer
from .sched import CosineLRScheduler
from .checkpoints import save_checkpoint, best_ckpt_path, last_ckpt_path


@dataclass
class TorchTrainArtifacts:
    best_path: Path
    last_path: Path
    history_path: Path


def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    out["images"] = batch["images"].to(device)
    # targets: list[dict]
    targets = []
    for t in batch["targets"]:
        targets.append({
            "boxes": t["boxes"].to(device),
            "labels": t["labels"].to(device),
        })
    out["targets"] = targets
    out["paths"] = batch["paths"]
    return out


def make_loaders(dataset_cfg_path: Path, split_manifest_path: Path, img_size: int, batch_size: int, num_workers: int):
    dcfg = load_dataset_config(dataset_cfg_path)
    manifest = load_split_manifest(split_manifest_path)

    train_ds = YoloDetectionDataset(
        dataset_cfg=dcfg,
        split_manifest=manifest,
        mode="train",
        transform=build_train_transforms(img_size),
    )
    val_ds = YoloDetectionDataset(
        dataset_cfg=dcfg,
        split_manifest=manifest,
        mode="val",
        transform=build_eval_transforms(img_size),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=detection_collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=detection_collate_fn,
        drop_last=False,
    )
    return train_loader, val_loader


@torch.no_grad()
def _val_loss(
    model: torch.nn.Module,
    loss_fn: Any,
    loader: DataLoader,
    device: torch.device,
    strides: Optional[List[int]] = None,
) -> float:
    model.eval()
    losses = []
    for batch in loader:
        b = _to_device(batch, device)
        out = model(b["images"])
        if hasattr(out, "preds"):  # YOLO-MvX HeadOut
            res = loss_fn(out.preds, out.strides, b["targets"])
            losses.append(float(res["loss"].item()))
        else:
            # DETR/Swin toy outputs: do not have a rigorous loss yet
            # For stability, just compute a proxy: mean abs logits
            logits = out["pred_logits"]
            losses.append(float(logits.abs().mean().item()))
    return float(np.mean(losses)) if losses else float("inf")


def train_torch_detector(
    *,
    model_name: str,
    model_cfg: Dict[str, Any],
    dataset_cfg_path: Path,
    split_manifest_path: Path,
    run_dir: Path,
    device_str: str,
    epochs: int,
    batch_size: int,
    img_size: int,
    lr: float,
    weight_decay: float,
    num_workers: int,
    seed: int,
) -> TorchTrainArtifacts:
    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")

    train_loader, val_loader = make_loaders(dataset_cfg_path, split_manifest_path, img_size, batch_size, num_workers)

    # Build model
    if model_name.lower() == "yolomvx":
        model = YoloMvX.from_config(model_cfg.get("yolomvx", model_cfg))
        loss_fn = YoloMvXLoss(num_classes=int(model_cfg.get("num_classes", 7)))
    else:
        # torch baselines: detr, swin_t
        model = build_baseline(model_name, model_cfg)
        loss_fn = None

    model = model.to(device)

    opt_name = str(model_cfg.get("train", {}).get("optimizer", "adamw"))
    optimizer = build_optimizer(model.parameters(), lr=lr, weight_decay=weight_decay, optim_name=opt_name)
    scheduler = CosineLRScheduler(
        optimizer,
        max_epochs=epochs,
        warmup_epochs=int(model_cfg.get("train", {}).get("warmup_epochs", 0)),
        min_lr_ratio=float(model_cfg.get("train", {}).get("min_lr_ratio", 0.05)),
    )

    best_loss = float("inf")
    history = {"epoch": [], "train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        epoch_losses = []

        for batch in train_loader:
            b = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            out = model(b["images"])

            if model_name.lower() == "yolomvx":
                res = loss_fn(out.preds, out.strides, b["targets"])
                loss = res["loss"]
            else:
                # Proxy objective for toy baselines (so training is defined)
                logits = out["pred_logits"]
                loss = logits.abs().mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(model_cfg.get("train", {}).get("grad_clip", 10.0)))
            optimizer.step()

            epoch_losses.append(float(loss.item()))

        scheduler.step(epoch)

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("inf")
        val_loss = _val_loss(model, loss_fn, val_loader, device)

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # checkpoints
        last_payload = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "model_name": model_name,
        }
        save_checkpoint(last_ckpt_path(run_dir), last_payload)

        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(best_ckpt_path(run_dir), last_payload)

    # Save learning curve history for reports
    hist_path = run_dir / "history.json"
    hist_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    return TorchTrainArtifacts(
        best_path=best_ckpt_path(run_dir),
        last_path=last_ckpt_path(run_dir),
        history_path=hist_path,
    )