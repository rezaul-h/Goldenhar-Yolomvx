from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from ..eval.config import load_yaml


@dataclass(frozen=True)
class TrainConfig:
    model_name: str
    dataset: str
    split: str
    run: int
    seed: int

    img_size: int
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float

    device: str
    num_workers: int

    # file paths
    model_cfg_path: Path
    dataset_cfg_path: Path
    split_manifest_path: Path

    # outputs
    output_root: Path


def _req(d: Dict[str, Any], k: str) -> Any:
    if k not in d:
        raise KeyError(f"Missing required key '{k}' in config.")
    return d[k]


def load_train_config(
    model_cfg_path: str | Path,
    dataset_cfg_path: str | Path,
    split_manifest_path: str | Path,
    run: int,
    seed: int,
    output_root: str | Path,
) -> TrainConfig:
    model_cfg_path = Path(model_cfg_path)
    dataset_cfg_path = Path(dataset_cfg_path)
    split_manifest_path = Path(split_manifest_path)
    output_root = Path(output_root)

    m = load_yaml(model_cfg_path)
    d = load_yaml(dataset_cfg_path)
    s = load_yaml(split_manifest_path)

    model_name = str(_req(m, "model"))
    dataset = str(_req(d["dataset"], "name"))
    split = str(_req(s, "split_name"))

    train = m.get("train", {})
    img_size = int(train.get("img_size", d["dataset"].get("img_size", 640)))
    epochs = int(train.get("epochs", 200))
    batch = int(train.get("batch_size", 16))
    lr = float(train.get("lr", 1e-3))
    wd = float(train.get("weight_decay", 5e-4))

    runtime = m.get("runtime", {})
    device = str(runtime.get("device", "cuda" if runtime.get("prefer_cuda", True) else "cpu"))
    num_workers = int(runtime.get("num_workers", 4))

    return TrainConfig(
        model_name=model_name,
        dataset=dataset,
        split=split,
        run=int(run),
        seed=int(seed),
        img_size=img_size,
        epochs=epochs,
        batch_size=batch,
        lr=lr,
        weight_decay=wd,
        device=device,
        num_workers=num_workers,
        model_cfg_path=model_cfg_path,
        dataset_cfg_path=dataset_cfg_path,
        split_manifest_path=split_manifest_path,
        output_root=output_root,
    )