from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, runtime_checkable


@dataclass(frozen=True)
class BaselineSpec:
    """
    Minimal baseline metadata used by trainers/evaluators.
    """
    name: str
    family: str  # "yolo" | "transformer"
    num_classes: int
    img_size: int
    pretrained: bool = False
    weights_path: Optional[str] = None


@runtime_checkable
class BaselineAdapter(Protocol):
    """
    Adapter interface for non-PyTorch baselines (e.g., Ultralytics CLI).
    """

    spec: BaselineSpec

    def train(self, *, run_dir: Path, data_yaml: Path, seed: int, overrides: Optional[Dict[str, Any]] = None) -> Path:
        """
        Train the model and return the best checkpoint path.
        """
        ...

    def predict(self, *, checkpoint: Path, images_dir: Path, out_dir: Path, conf: float = 0.25) -> Path:
        """
        Run inference and write predictions to out_dir.
        Returns a path to a prediction artifact (e.g., predictions.json/csv).
        """
        ...