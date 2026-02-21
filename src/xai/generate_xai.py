from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from .config import load_yaml
from .heatmaps import to_numpy_img, overlay_heatmap, save_image
from .gradcam import GradCAM, GradCAMpp
from .occlusion import OcclusionSensitivity
from .detector_interface import DetectionTarget, default_detector_score_fn

from ..models.yolomvx import build_yolomvx
from ..models.baselines import build_baseline


def _load_image(path: Path, img_size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if img_size > 0:
        img = img.resize((img_size, img_size))
    x = torch.from_numpy((torch.tensor(list(img.getdata())).view(img.size[1], img.size[0], 3).numpy())).float()
    x = x / 255.0
    x = x.permute(2, 0, 1).contiguous()  # (3,H,W)
    return x


def _resolve_target_layer(model: torch.nn.Module, name: str) -> torch.nn.Module:
    """
    Resolve a target layer by dotted path, e.g.:
      - backbone.stage5
      - backbone.stage5.0
      - backbone.sppf
    """
    cur = model
    for part in name.split("."):
        if part.isdigit():
            cur = cur[int(part)]  # type: ignore[index]
        else:
            cur = getattr(cur, part)
    if not isinstance(cur, torch.nn.Module):
        raise TypeError("Resolved target layer is not a nn.Module.")
    return cur


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate XAI overlays for a folder of images.")
    ap.add_argument("--config", type=str, required=True, help="configs/xai/xai.yaml")
    ap.add_argument("--weights", type=str, required=False, default=None, help="optional torch checkpoint (state_dict)")
    ap.add_argument("--images", type=str, required=True, help="folder of images")
    ap.add_argument("--outdir", type=str, required=True, help="output folder")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    model_name = str(cfg.get("model", "yolomvx")).lower()
    img_size = int(cfg.get("img_size", 640))
    device = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # Build model
    if model_name == "yolomvx":
        model = build_yolomvx(cfg.get("yolomvx", cfg))
    else:
        model = build_baseline(model_name, cfg)

    if isinstance(model, torch.nn.Module):
        model = model.to(device).eval()
    else:
        raise TypeError("XAI generation expects a torch.nn.Module model (not an adapter).")

    # Optional weights loading (state_dict)
    if args.weights is not None:
        ckpt = torch.load(args.weights, map_location="cpu")
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            state = ckpt["model_state"]
        elif isinstance(ckpt, dict):
            state = ckpt
        else:
            raise ValueError("Unsupported checkpoint format for XAI.")
        model.load_state_dict(state, strict=False)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    method = str(cfg.get("method", "gradcam")).lower()
    alpha = float(cfg.get("alpha", 0.45))

    # target specification for detectors
    cls_id = int(cfg.get("target", {}).get("cls_id", 0))
    det_index = cfg.get("target", {}).get("det_index", None)
    det_index = int(det_index) if det_index is not None else None
    target = DetectionTarget(cls_id=cls_id, det_index=det_index)

    # choose method
    cam_obj = None
    occ_obj = None

    if method in {"gradcam", "gradcam++"}:
        layer_name = str(cfg.get("gradcam", {}).get("target_layer", "backbone.stage5"))
        layer = _resolve_target_layer(model, layer_name)
        cam_obj = GradCAMpp(model, layer) if method == "gradcam++" else GradCAM(model, layer)

    if method == "occlusion":
        occ = cfg.get("occlusion", {})
        occ_obj = OcclusionSensitivity(
            patch=int(occ.get("patch", 32)),
            stride=int(occ.get("stride", 16)),
            fill=float(occ.get("fill", 0.0)),
        )

    img_dir = Path(args.images)
    paths = sorted([p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}])

    for p in paths:
        x = _load_image(p, img_size=img_size).unsqueeze(0).to(device)  # (1,3,H,W)
        img_rgb = to_numpy_img(x[0])

        if method in {"gradcam", "gradcam++"}:
            # forward + score
            score = default_detector_score_fn(model, x, target)
            out = cam_obj(x, score, out_hw=(x.shape[2], x.shape[3]))  # type: ignore[misc]
            hm = out.heatmap.detach().cpu().numpy()
        elif method == "occlusion":
            out = occ_obj(model, x, lambda m, xx: default_detector_score_fn(m, xx, target))  # type: ignore[misc]
            hm = out.heatmap.detach().cpu().numpy()
        else:
            raise ValueError(f"Unsupported method: {method}")

        overlay = overlay_heatmap(img_rgb, hm, alpha=alpha)
        save_image(overlay, outdir / f"{p.stem}_{method}.png")

    # cleanup
    if cam_obj is not None:
        cam_obj.close()

    print(f"[OK] Saved XAI overlays to: {outdir.resolve()}")


if __name__ == "__main__":
    main()