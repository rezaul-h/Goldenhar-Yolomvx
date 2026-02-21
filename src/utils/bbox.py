from __future__ import annotations

from typing import List, Tuple

import torch


def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """
    boxes: (...,4) [x1,y1,x2,y2]
    returns: (...,4) [cx,cy,w,h]
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    w = (x2 - x1).clamp(min=0)
    h = (y2 - y1).clamp(min=0)
    return torch.stack([cx, cy, w, h], dim=-1)


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    boxes: (...,4) [cx,cy,w,h]
    returns: (...,4) [x1,y1,x2,y2]
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_iou_xyxy(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    IoU between box sets in xyxy.
    box1: (N,4), box2: (M,4) -> (N,M)
    """
    b1 = box1[:, None, :]  # (N,1,4)
    b2 = box2[None, :, :]  # (1,M,4)

    ix1 = torch.max(b1[..., 0], b2[..., 0])
    iy1 = torch.max(b1[..., 1], b2[..., 1])
    ix2 = torch.min(b1[..., 2], b2[..., 2])
    iy2 = torch.min(b1[..., 3], b2[..., 3])

    iw = (ix2 - ix1).clamp(min=0)
    ih = (iy2 - iy1).clamp(min=0)
    inter = iw * ih

    a1 = (b1[..., 2] - b1[..., 0]).clamp(min=0) * (b1[..., 3] - b1[..., 1]).clamp(min=0)
    a2 = (b2[..., 2] - b2[..., 0]).clamp(min=0) * (b2[..., 3] - b2[..., 1]).clamp(min=0)
    union = a1 + a2 - inter + eps

    return inter / union


def nms_xyxy(boxes: torch.Tensor, scores: torch.Tensor, iou_th: float = 0.5) -> torch.Tensor:
    """
    Pure PyTorch NMS for xyxy boxes.
    boxes: (N,4), scores: (N,)
    returns: indices kept (K,)
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    x1, y1, x2, y2 = boxes.unbind(-1)
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0:
        i = int(order[0])
        keep.append(i)
        if order.numel() == 1:
            break

        rest = order[1:]
        xx1 = torch.maximum(x1[i], x1[rest])
        yy1 = torch.maximum(y1[i], y1[rest])
        xx2 = torch.minimum(x2[i], x2[rest])
        yy2 = torch.minimum(y2[i], y2[rest])

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h

        iou = inter / (areas[i] + areas[rest] - inter + 1e-7)
        mask = iou <= float(iou_th)
        order = rest[mask]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)