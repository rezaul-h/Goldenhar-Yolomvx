from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def bbox_ciou(box1_xyxy: torch.Tensor, box2_xyxy: torch.Tensor) -> torch.Tensor:
    """
    CIoU for xyxy boxes.
    box1, box2: (..., 4)
    """
    x1, y1, x2, y2 = box1_xyxy.unbind(-1)
    x1g, y1g, x2g, y2g = box2_xyxy.unbind(-1)

    # intersection
    ix1 = torch.max(x1, x1g)
    iy1 = torch.max(y1, y1g)
    ix2 = torch.min(x2, x2g)
    iy2 = torch.min(y2, y2g)
    iw = (ix2 - ix1).clamp(min=0)
    ih = (iy2 - iy1).clamp(min=0)
    inter = iw * ih

    # union
    a1 = ((x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0))
    a2 = ((x2g - x1g).clamp(min=0) * (y2g - y1g).clamp(min=0))
    union = a1 + a2 - inter + 1e-7
    iou = inter / union

    # center distance
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    cxg = (x1g + x2g) * 0.5
    cyg = (y1g + y2g) * 0.5
    rho2 = (cx - cxg) ** 2 + (cy - cyg) ** 2

    # enclosing box diagonal
    ex1 = torch.min(x1, x1g)
    ey1 = torch.min(y1, y1g)
    ex2 = torch.max(x2, x2g)
    ey2 = torch.max(y2, y2g)
    c2 = ((ex2 - ex1) ** 2 + (ey2 - ey1) ** 2) + 1e-7

    # aspect ratio term
    w1 = (x2 - x1).clamp(min=1e-7)
    h1 = (y2 - y1).clamp(min=1e-7)
    w2 = (x2g - x1g).clamp(min=1e-7)
    h2 = (y2g - y1g).clamp(min=1e-7)
    v = (4.0 / (3.14159265 ** 2)) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)) ** 2
    with torch.no_grad():
        alpha = v / (1.0 - iou + v + 1e-7)

    ciou = iou - (rho2 / c2) - alpha * v
    return ciou


def decode_boxes(
    raw_box: torch.Tensor, grid_xy: torch.Tensor, stride: int
) -> torch.Tensor:
    """
    raw_box: (N, 4) where [tx,ty,tw,th] are unconstrained
    grid_xy: (N, 2) grid cell centers in feature coords
    stride:  int
    Returns xyxy in pixels.
    """
    # center offsets
    xy = (raw_box[:, 0:2].sigmoid() + grid_xy) * stride
    wh = raw_box[:, 2:4].exp() * stride
    x1y1 = xy - wh * 0.5
    x2y2 = xy + wh * 0.5
    return torch.cat([x1y1, x2y2], dim=1)


@dataclass(frozen=True)
class LossWeights:
    box: float = 5.0
    obj: float = 1.0
    cls: float = 1.0


class YoloMvXLoss(nn.Module):
    """
    A minimal, working loss with simple assignment:
      - for each GT, pick the best-matching prediction cell across all scales
        using distance-to-center + size prior (approx).
    This is NOT SimOTA/TaskAligned; it is intentionally simple and stable.
    """
    def __init__(self, num_classes: int, weights: LossWeights | None = None):
        super().__init__()
        self.num_classes = num_classes
        self.w = weights if weights is not None else LossWeights()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    @staticmethod
    def _make_grids(preds: List[torch.Tensor], strides: List[int], device: torch.device):
        grids = []
        for p, s in zip(preds, strides):
            # p: (B,1,H,W,K)
            H, W = p.shape[2], p.shape[3]
            y, x = torch.meshgrid(
                torch.arange(H, device=device), torch.arange(W, device=device),
                indexing="ij"
            )
            grid = torch.stack([x, y], dim=-1).view(-1, 2).float()  # (HW,2)
            grids.append((grid, s, H, W))
        return grids

    def forward(self, head_preds: List[torch.Tensor], strides: List[int], targets: List[Dict[str, torch.Tensor]]):
        """
        head_preds: list of (B,1,H,W,K)
        targets: list (len B) each dict with:
          - boxes: (Ni,4) xyxy pixels
          - labels:(Ni,)
        """
        device = head_preds[0].device
        B = head_preds[0].shape[0]
        grids = self._make_grids(head_preds, strides, device=device)

        total_box = torch.tensor(0.0, device=device)
        total_obj = torch.tensor(0.0, device=device)
        total_cls = torch.tensor(0.0, device=device)
        n_pos = 0

        for bi in range(B):
            # gather all predictions across scales into one list
            pred_all = []
            meta_all = []
            for si, (p, (grid, s, H, W)) in enumerate(zip(head_preds, grids)):
                # (1, H, W, K) -> (HW, K)
                pi = p[bi, 0].view(-1, p.shape[-1])
                pred_all.append(pi)
                meta_all.append((grid, s))
            pred_all = torch.cat(pred_all, dim=0)  # (M,K)

            raw_box = pred_all[:, 0:4]
            raw_obj = pred_all[:, 4:5]
            raw_cls = pred_all[:, 5:]

            # build global grid and stride vectors
            grid_cat = torch.cat([m[0] for m in meta_all], dim=0)  # (M,2)
            stride_cat = torch.cat(
                [torch.full((m[0].shape[0], 1), float(m[1]), device=device) for m in meta_all], dim=0
            )  # (M,1)

            # decode boxes
            # decode uses per-row stride -> handle by decoding per stride group
            # (simple): decode with stride_cat scalar per row
            # approximate by using stride_cat in xy/wh scaling
            xy = (raw_box[:, 0:2].sigmoid() + grid_cat) * stride_cat
            wh = raw_box[:, 2:4].exp() * stride_cat
            box_xyxy = torch.cat([xy - 0.5 * wh, xy + 0.5 * wh], dim=1)

            gt_boxes = targets[bi]["boxes"].to(device)
            gt_labels = targets[bi]["labels"].to(device)

            M = pred_all.shape[0]
            obj_t = torch.zeros((M, 1), device=device)
            cls_t = torch.zeros((M, self.num_classes), device=device)

            if gt_boxes.numel() == 0:
                # no objects: only objectness loss to zeros
                total_obj = total_obj + self.bce(raw_obj, obj_t).mean()
                continue

            # assignment: each gt picks best pred index
            # score = -center_dist - 0.5*size_dist + iou
            gt_centers = 0.5 * (gt_boxes[:, 0:2] + gt_boxes[:, 2:4])  # (N,2)
            pred_centers = 0.5 * (box_xyxy[:, 0:2] + box_xyxy[:, 2:4])  # (M,2)

            # center distance
            d = torch.cdist(gt_centers, pred_centers)  # (N,M)

            # size distance
            gt_wh = (gt_boxes[:, 2:4] - gt_boxes[:, 0:2]).clamp(min=1.0)
            pr_wh = (box_xyxy[:, 2:4] - box_xyxy[:, 0:2]).clamp(min=1.0)
            # (N,M,2) -> (N,M)
            sd = torch.cdist(gt_wh, pr_wh)

            # IoU matrix (N,M) (vectorized enough for modest sizes)
            # compute iou by broadcasting
            gb = gt_boxes[:, None, :]  # (N,1,4)
            pb = box_xyxy[None, :, :]  # (1,M,4)
            ix1 = torch.max(gb[..., 0], pb[..., 0])
            iy1 = torch.max(gb[..., 1], pb[..., 1])
            ix2 = torch.min(gb[..., 2], pb[..., 2])
            iy2 = torch.min(gb[..., 3], pb[..., 3])
            iw = (ix2 - ix1).clamp(min=0)
            ih = (iy2 - iy1).clamp(min=0)
            inter = iw * ih
            ga = ((gb[..., 2] - gb[..., 0]).clamp(min=0) * (gb[..., 3] - gb[..., 1]).clamp(min=0))
            pa = ((pb[..., 2] - pb[..., 0]).clamp(min=0) * (pb[..., 3] - pb[..., 1]).clamp(min=0))
            iou = inter / (ga + pa - inter + 1e-7)

            score = iou - (d / (d.mean() + 1e-7)) - 0.5 * (sd / (sd.mean() + 1e-7))
            best_idx = torch.argmax(score, dim=1)  # (N,)

            # set targets
            obj_t[best_idx] = 1.0
            cls_t[best_idx, gt_labels] = 1.0

            # losses
            # objectness on all
            total_obj = total_obj + self.bce(raw_obj, obj_t).mean()

            # classification on positives only
            pos = obj_t.squeeze(1) > 0.5
            if pos.any():
                n_pos += int(pos.sum().item())
                total_cls = total_cls + self.bce(raw_cls[pos], cls_t[pos]).mean()

                # box loss on positives: CIoU
                pred_pos = box_xyxy[pos]
                gt_pos = gt_boxes[torch.arange(gt_boxes.shape[0], device=device)]
                # align by gt order -> use assigned
                gt_pos = gt_boxes[torch.arange(gt_boxes.shape[0], device=device)]
                gt_pos = gt_boxes  # (N,4)
                pred_for_gt = box_xyxy[best_idx]  # (N,4)
                ciou = bbox_ciou(pred_for_gt, gt_pos)
                total_box = total_box + (1.0 - ciou).mean()

        # normalize by batch (and avoid zero division)
        denom = max(B, 1)
        loss = self.w.box * (total_box / denom) + self.w.obj * (total_obj / denom) + self.w.cls * (total_cls / denom)
        return {
            "loss": loss,
            "loss_box": (total_box / denom).detach(),
            "loss_obj": (total_obj / denom).detach(),
            "loss_cls": (total_cls / denom).detach(),
            "n_pos": torch.tensor(float(n_pos), device=device),
        }