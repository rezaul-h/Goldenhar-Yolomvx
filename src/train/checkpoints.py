from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch


def run_dir(output_root: Path, model: str, dataset: str, split: str, run: int) -> Path:
    p = output_root / "checkpoints" / model / dataset / split / f"run{run}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_checkpoint(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, map_location: str = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location)


def best_ckpt_path(run_dir: Path) -> Path:
    return run_dir / "best.pt"


def last_ckpt_path(run_dir: Path) -> Path:
    return run_dir / "last.pt"