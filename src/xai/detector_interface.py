from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass(frozen=True)
class DetectionTarget:
    """
    Defines what to explain for a detector.
    You can target:
      - a specific class id (cls_id)
      - optionally a specific detection index after NMS (det_index)
    """
    cls_id: int
    det_index: Optional[int] = None


def default_detector_score_fn(
    model,
    x: torch.Tensor,
    target: DetectionTarget,
) -> torch.Tensor:
    """
    Generic scoring function for detectors that output:
      out["pred_logits"] and out["pred_boxes"] (DETR/Swin baseline),
    OR YOLO-like raw head output (HeadOut) where you later implement decode.

    For DETR-like:
      score = max softmax prob of target class across queries.

    NOTE: For YOLO-MvX raw output, you should add a decode step later (NMS + class conf)
    and then compute score of a selected detection. For now, this function raises.
    """
    out = model(x)

    if isinstance(out, dict) and "pred_logits" in out:
        logits = out["pred_logits"]  # (B,Q,C+1)
        probs = logits.softmax(dim=-1)
        # target class prob across queries
        cls_prob = probs[..., target.cls_id]  # (B,Q)
        if target.det_index is None:
            return cls_prob.max()
        return cls_prob[:, target.det_index].mean()

    # YOLO-MvX returns HeadOut with preds list
    if hasattr(out, "preds"):
        raise NotImplementedError(
            "YOLO-MvX scoring requires decoding predictions to detections.\n"
            "Implement a decode+NMS and then compute confidence for target class/detection."
        )

    raise TypeError("Unsupported model output for default_detector_score_fn.")