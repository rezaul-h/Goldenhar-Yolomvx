from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .base import BaselineSpec


@dataclass
class UltralyticsYOLOAdapter:
    """
    Thin adapter for YOLO baselines that delegates training/inference to Ultralytics CLI.

    Requirements:
      pip install ultralytics

    Notes:
      - This adapter does not implement YOLOv9/10/11 architectures directly.
      - It provides consistent IO so your pipeline remains stable.
    """
    spec: BaselineSpec
    ultralytics_model: str  # e.g., 'yolov9c.pt' or 'yolov10n.pt' etc.

    @staticmethod
    def from_config(name: str, cfg: Dict[str, Any]) -> "UltralyticsYOLOAdapter":
        num_classes = int(cfg.get("num_classes", 7))
        img_size = int(cfg.get("img_size", 640))
        pretrained = bool(cfg.get("pretrained", True))
        weights_path = cfg.get("weights_path", None)

        # Conservative defaults; replace with exact weights you use.
        default_weights = {
            "yolov9": "yolov9c.pt",
            "yolov10": "yolov10n.pt",
            "yolov11": "yolo11n.pt",
        }
        model_weights = str(weights_path) if weights_path else default_weights.get(name, "yolov8n.pt")

        return UltralyticsYOLOAdapter(
            spec=BaselineSpec(
                name=name,
                family="yolo",
                num_classes=num_classes,
                img_size=img_size,
                pretrained=pretrained,
                weights_path=weights_path,
            ),
            ultralytics_model=model_weights,
        )

    def _run(self, cmd: list[str], env: Optional[Dict[str, str]] = None) -> None:
        e = os.environ.copy()
        if env:
            e.update(env)
        proc = subprocess.run(cmd, env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}")
        # Print is acceptable for CLI scripts; upstream can redirect to logs.
        print(proc.stdout)

    def train(
        self,
        *,
        run_dir: Path,
        data_yaml: Path,
        seed: int,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Path:
        run_dir.mkdir(parents=True, exist_ok=True)
        overrides = overrides or {}

        imgsz = int(overrides.get("imgsz", self.spec.img_size))
        epochs = int(overrides.get("epochs", 200))
        batch = int(overrides.get("batch", 16))
        lr0 = float(overrides.get("lr0", 0.01))
        device = str(overrides.get("device", "0"))

        # Ultralytics uses project/name to control output directory.
        project = str(run_dir)
        name = "train"

        cmd = [
            "yolo",
            "detect",
            "train",
            f"model={self.ultralytics_model}",
            f"data={str(data_yaml)}",
            f"imgsz={imgsz}",
            f"epochs={epochs}",
            f"batch={batch}",
            f"lr0={lr0}",
            f"seed={int(seed)}",
            f"device={device}",
            f"project={project}",
            f"name={name}",
            "exist_ok=True",
        ]

        self._run(cmd)

        # Ultralytics default best checkpoint path
        best = run_dir / "train" / "weights" / "best.pt"
        if not best.exists():
            # fallback: last.pt
            last = run_dir / "train" / "weights" / "last.pt"
            if last.exists():
                return last
            raise FileNotFoundError(f"Could not find trained weights at: {best} (or last.pt).")
        return best

    def predict(
        self,
        *,
        checkpoint: Path,
        images_dir: Path,
        out_dir: Path,
        conf: float = 0.25,
    ) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        project = str(out_dir)
        name = "pred"

        cmd = [
            "yolo",
            "detect",
            "predict",
            f"model={str(checkpoint)}",
            f"source={str(images_dir)}",
            f"conf={float(conf)}",
            f"project={project}",
            f"name={name}",
            "exist_ok=True",
            "save_txt=True",
            "save_conf=True",
        ]
        self._run(cmd)

        # For downstream error analysis, we provide a lightweight manifest
        artifact = out_dir / "pred_manifest.json"
        payload = {
            "checkpoint": str(checkpoint),
            "images_dir": str(images_dir),
            "pred_dir": str(out_dir / "pred"),
            "format": "ultralytics_txt_with_conf",
        }
        artifact.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return artifact