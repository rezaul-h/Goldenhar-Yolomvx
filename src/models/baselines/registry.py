from __future__ import annotations

from typing import Any, Callable, Dict

from .yolo_adapter import UltralyticsYOLOAdapter
from .detr import build_detr_tiny
from .swin_t import build_swin_tiny

# Registry returns either torch.nn.Module OR an adapter object.
BASELINE_REGISTRY: Dict[str, Callable[[Dict[str, Any]], Any]] = {
    "yolov9": lambda cfg: UltralyticsYOLOAdapter.from_config("yolov9", cfg),
    "yolov10": lambda cfg: UltralyticsYOLOAdapter.from_config("yolov10", cfg),
    "yolov11": lambda cfg: UltralyticsYOLOAdapter.from_config("yolov11", cfg),
    "detr": lambda cfg: build_detr_tiny(cfg),
    "swin_t": lambda cfg: build_swin_tiny(cfg),
}


def build_baseline(name: str, cfg: Dict[str, Any]) -> Any:
    name = name.lower().strip()
    if name not in BASELINE_REGISTRY:
        raise KeyError(f"Unknown baseline '{name}'. Available: {sorted(BASELINE_REGISTRY.keys())}")
    return BASELINE_REGISTRY[name](cfg)