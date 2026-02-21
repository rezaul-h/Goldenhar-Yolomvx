# src/xai/cams.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hooks import ActivationGradientHook
from .heatmaps import normalize_map, resize_map


@dataclass(frozen=True)
class CAMResult:
    """
    Standard output container for CAM methods.
    """
    heatmap: torch.Tensor            # (H,W) in [0,1]
    score: float                     # scalar score used for attribution
    method: str                      # "cam"|"gradcam"|"gradcam++"|"scorecam"
    layer_name: str                  # human-readable layer identifier


def _resolve_module_by_name(model: nn.Module, dotted_path: str) -> nn.Module:
    """
    Resolve a target module by dotted path, e.g.:
      "backbone.stage5"
      "neck.pan5"
      "backbone.stage5.m.0"
    """
    cur: Any = model
    for part in dotted_path.split("."):
        if part.isdigit():
            cur = cur[int(part)]
        else:
            cur = getattr(cur, part)
    if not isinstance(cur, nn.Module):
        raise TypeError(f"Resolved object is not a nn.Module: {dotted_path}")
    return cur


def _default_score_from_logits(logits: torch.Tensor, class_id: Optional[int] = None) -> torch.Tensor:
    """
    logits: (1,C) or (C,)
    class_id:
      - if provided, uses logits[class_id]
      - else uses max logit
    Returns a scalar tensor.
    """
    if logits.dim() == 2:
        logits = logits[0]
    if class_id is None:
        return logits.max()
    return logits[int(class_id)]


class BaseCAM:
    """
    Base interface:
      - attach hooks to capture activations (+ optionally gradients)
      - compute a heatmap in [0,1] resized to out_hw
    """

    def __init__(self, model: nn.Module, target_layer: Union[str, nn.Module], layer_name: Optional[str] = None):
        self.model = model
        if isinstance(target_layer, str):
            self.layer_name = target_layer
            self.layer = _resolve_module_by_name(model, target_layer)
        else:
            self.layer = target_layer
            self.layer_name = layer_name or target_layer.__class__.__name__

    def close(self) -> None:
        pass

    def __call__(
        self,
        x: torch.Tensor,
        score: torch.Tensor,
        out_hw: Optional[Tuple[int, int]] = None,
    ) -> CAMResult:
        raise NotImplementedError


class CAM(BaseCAM):
    """
    Classic CAM requires a linear classifier and uses class weights.
    This implementation supports models where you can provide:
      - `classifier_weight`: Tensor (C, K) aligned with channels K of target layer activation
    or the model exposes `fc.weight` / `classifier.weight`.

    Usage (classification):
      logits = model(x)  # (1,C)
      score = logits[0, cls]
      cam = CAM(model, "features.layer4", classifier_weight=model.fc.weight)
      res = cam(x, score)
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Union[str, nn.Module],
        classifier_weight: Optional[torch.Tensor] = None,
        layer_name: Optional[str] = None,
    ):
        super().__init__(model, target_layer, layer_name=layer_name)
        self.hook = ActivationGradientHook(self.layer).attach()

        self.classifier_weight = classifier_weight
        if self.classifier_weight is None:
            # common attribute names
            for name in ["fc", "classifier", "head"]:
                m = getattr(model, name, None)
                if isinstance(m, nn.Linear):
                    self.classifier_weight = m.weight
                    break

        if self.classifier_weight is None:
            raise ValueError(
                "CAM requires classifier weights (C,K). Provide `classifier_weight=` or expose model.fc/classifier."
            )

    def close(self) -> None:
        self.hook.remove()

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, score: torch.Tensor, out_hw: Optional[Tuple[int, int]] = None) -> CAMResult:
        _ = self.model(x)  # ensure activation is populated
        act = self.hook.out.activation
        if act is None:
            raise RuntimeError("No activation captured. Ensure target_layer participates in forward pass.")

        # act: (1,K,h,w)
        if out_hw is None:
            out_hw = (x.shape[2], x.shape[3])

        # Identify class index from the provided score if possible is not reliable;
        # thus CAM expects the caller to pass the class id separately via score construction.
        # We approximate by using gradient-free CAM with the *max* class weight if unknown.
        # Better: use the helper `make_score_fn` below.
        # Here we choose max-weighted CAM using argmax of logits proxy:
        # For strictness, we compute weights by the argmax over a forward pass.
        logits = self.model(x)
        if isinstance(logits, dict) and "logits" in logits:
            logits = logits["logits"]
        if not isinstance(logits, torch.Tensor):
            raise TypeError("CAM expects classification logits as a Tensor output (or dict with key 'logits').")

        cls = int(logits[0].argmax().item())
        w = self.classifier_weight[cls]  # (K,)

        cam = (act[0] * w.view(-1, 1, 1)).sum(dim=0)  # (h,w)
        cam = F.relu(cam)
        cam = normalize_map(cam)
        cam = resize_map(cam, out_hw)
        cam = normalize_map(cam)

        return CAMResult(
            heatmap=cam,
            score=float(score.detach().cpu().item()),
            method="cam",
            layer_name=self.layer_name,
        )


class GradCAM(BaseCAM):
    """
    Grad-CAM for CNN-like feature maps.
    Works for classification or any scalar score that depends on the network output.
    """

    def __init__(self, model: nn.Module, target_layer: Union[str, nn.Module], layer_name: Optional[str] = None):
        super().__init__(model, target_layer, layer_name=layer_name)
        self.hook = ActivationGradientHook(self.layer).attach()

    def close(self) -> None:
        self.hook.remove()

    def __call__(self, x: torch.Tensor, score: torch.Tensor, out_hw: Optional[Tuple[int, int]] = None) -> CAMResult:
        self.model.zero_grad(set_to_none=True)
        self.hook.clear()

        # caller may or may not have run forward; to be safe:
        _ = self.model(x)

        score.backward(retain_graph=True)

        act = self.hook.out.activation
        grad = self.hook.out.gradient
        if act is None or grad is None:
            raise RuntimeError("No activation/gradient captured. Ensure target_layer participates in forward pass.")

        # weights: GAP over gradients
        w = grad.mean(dim=(2, 3), keepdim=True)  # (1,C,1,1)
        cam = (w * act).sum(dim=1, keepdim=True)  # (1,1,h,w)
        cam = F.relu(cam)
        cam = normalize_map(cam)[0, 0]

        if out_hw is None:
            out_hw = (x.shape[2], x.shape[3])
        cam = resize_map(cam, out_hw)
        cam = normalize_map(cam)

        return CAMResult(
            heatmap=cam,
            score=float(score.detach().cpu().item()),
            method="gradcam",
            layer_name=self.layer_name,
        )


class GradCAMpp(BaseCAM):
    """
    Grad-CAM++: improved localization via higher-order gradient weighting.
    """

    def __init__(self, model: nn.Module, target_layer: Union[str, nn.Module], layer_name: Optional[str] = None):
        super().__init__(model, target_layer, layer_name=layer_name)
        self.hook = ActivationGradientHook(self.layer).attach()

    def close(self) -> None:
        self.hook.remove()

    def __call__(self, x: torch.Tensor, score: torch.Tensor, out_hw: Optional[Tuple[int, int]] = None) -> CAMResult:
        self.model.zero_grad(set_to_none=True)
        self.hook.clear()

        _ = self.model(x)
        score.backward(retain_graph=True)

        act = self.hook.out.activation
        grad = self.hook.out.gradient
        if act is None or grad is None:
            raise RuntimeError("No activation/gradient captured. Ensure target_layer participates in forward pass.")

        grad2 = grad.pow(2)
        grad3 = grad.pow(3)
        sum_act_grad3 = (act * grad3).sum(dim=(2, 3), keepdim=True)

        eps = 1e-7
        alpha = grad2 / (2.0 * grad2 + sum_act_grad3 + eps)
        alpha = torch.where(torch.isfinite(alpha), alpha, torch.zeros_like(alpha))

        w = (alpha * F.relu(grad)).sum(dim=(2, 3), keepdim=True)  # (1,C,1,1)
        cam = (w * act).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = normalize_map(cam)[0, 0]

        if out_hw is None:
            out_hw = (x.shape[2], x.shape[3])
        cam = resize_map(cam, out_hw)
        cam = normalize_map(cam)

        return CAMResult(
            heatmap=cam,
            score=float(score.detach().cpu().item()),
            method="gradcam++",
            layer_name=self.layer_name,
        )


class ScoreCAM(BaseCAM):
    """
    Score-CAM (gradient-free):
      - uses activation maps as masks to perturb input and measure score changes.

    This is slower but useful when gradients are noisy.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Union[str, nn.Module],
        score_fn: Optional[Callable[[Any], torch.Tensor]] = None,
        max_maps: int = 32,
        layer_name: Optional[str] = None,
    ):
        super().__init__(model, target_layer, layer_name=layer_name)
        self.hook = ActivationGradientHook(self.layer).attach()
        self.score_fn = score_fn  # maps model output -> scalar
        self.max_maps = int(max_maps)

    def close(self) -> None:
        self.hook.remove()

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, score: torch.Tensor, out_hw: Optional[Tuple[int, int]] = None) -> CAMResult:
        """
        Note: `score` is only used to report; ScoreCAM recomputes weights using score_fn.
        """
        if out_hw is None:
            out_hw = (x.shape[2], x.shape[3])

        out = self.model(x)
        act = self.hook.out.activation
        if act is None:
            raise RuntimeError("No activation captured. Ensure target_layer participates in forward pass.")

        if self.score_fn is None:
            # Default: if out is logits tensor (1,C) -> max logit
            if isinstance(out, torch.Tensor):
                score_fn = lambda o: o.max()
            elif isinstance(out, dict) and "logits" in out and isinstance(out["logits"], torch.Tensor):
                score_fn = lambda o: o["logits"].max()
            else:
                raise ValueError("Provide score_fn for non-standard model outputs.")
        else:
            score_fn = self.score_fn

        # act: (1,C,h,w)
        A = act[0]  # (C,h,w)
        C, h, w = A.shape

        # choose top activation maps by energy to reduce compute
        energy = A.flatten(1).abs().mean(dim=1)  # (C,)
        idx = torch.argsort(energy, descending=True)[: min(C, self.max_maps)]
        A = A[idx]  # (K,h,w)

        cam = torch.zeros((h, w), device=x.device)
        base_inp = x

        for k in range(A.shape[0]):
            m = A[k]
            m = normalize_map(m)
            m_up = resize_map(m, (x.shape[2], x.shape[3]))  # (H,W)
            masked = base_inp * m_up[None, None, :, :]
            s = score_fn(self.model(masked))
            cam += float(s.detach().cpu().item()) * m  # accumulate in feature resolution

        cam = F.relu(cam)
        cam = normalize_map(cam)
        cam = resize_map(cam, out_hw)
        cam = normalize_map(cam)

        return CAMResult(
            heatmap=cam,
            score=float(score.detach().cpu().item()),
            method="scorecam",
            layer_name=self.layer_name,
        )


# --------------------------- Helper: score construction ---------------------------

def make_classification_score(
    model: nn.Module,
    x: torch.Tensor,
    class_id: Optional[int] = None,
    logits_key: Optional[str] = None,
) -> torch.Tensor:
    """
    Convenience helper to obtain a scalar score for CAM methods.

    - If model(x) returns Tensor (1,C): uses class logit or max logit.
    - If model(x) returns dict: uses dict[logits_key] if provided; else tries 'logits' then 'pred_logits'.

    Returns scalar Tensor.
    """
    out = model(x)
    if isinstance(out, torch.Tensor):
        return _default_score_from_logits(out, class_id=class_id)

    if isinstance(out, dict):
        if logits_key is not None and logits_key in out and isinstance(out[logits_key], torch.Tensor):
            return _default_score_from_logits(out[logits_key], class_id=class_id)
        for k in ["logits", "pred_logits"]:
            if k in out and isinstance(out[k], torch.Tensor):
                # pred_logits may be (1,Q,C+1); take max over queries for a target class
                t = out[k]
                if t.dim() == 3:
                    # (1,Q,C) -> (C,) via max across queries
                    t2 = t[0].max(dim=0).values
                    return _default_score_from_logits(t2, class_id=class_id)
                return _default_score_from_logits(t, class_id=class_id)

    raise TypeError("Unsupported model output format for make_classification_score.")