from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from .config import ensure_dir, load_yaml
from .discovery import RunKey, discover_run_logs, candidate_efficiency_files


def _load_checkpoint(path: Path) -> Optional[torch.nn.Module]:
    """
    Generic loader:
      - if a full torch module is saved -> torch.load returns nn.Module
      - if state_dict -> can't reconstruct without model code (returns None)
    """
    try:
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, torch.nn.Module):
            return obj.eval()
        if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], torch.nn.Module):
            return obj["model"].eval()
        # state_dict only: not enough to instantiate
        return None
    except Exception:
        return None


def count_params(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def measure_latency_ms(
    model: torch.nn.Module,
    img_size: int,
    device: str = "cpu",
    warmup: int = 10,
    iters: int = 50,
) -> float:
    model = model.to(device).eval()
    x = torch.randn(1, 3, img_size, img_size, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)

    # Timed
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            _ = model(x)
    t1 = time.perf_counter()

    return float((t1 - t0) * 1000.0 / iters)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute or collect efficiency metrics (params, latency).")
    ap.add_argument("--config", type=str, required=True, help="configs/eval/efficiency.yaml")
    ap.add_argument("--output-root", type=str, required=True, help="outputs/")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    output_root = Path(args.output_root)

    out_raw = ensure_dir(output_root / "metrics" / "raw" / "efficiency")
    out_agg = ensure_dir(output_root / "metrics" / "aggregated")

    # If you already export efficiency JSON files, we only collect them.
    # If not, we can *optionally* compute from checkpoints (only works if checkpoints serialize the full nn.Module).
    runs = discover_run_logs(output_root)
    if not runs:
        raise FileNotFoundError(f"No runs discovered under {output_root / 'logs'}.")

    # Collection first
    collected_rows: List[Dict[str, Any]] = []
    missing: List[RunKey] = []
    for rk in runs:
        files = candidate_efficiency_files(output_root, rk)
        if not files:
            missing.append(rk)
            continue
        p = sorted(files)[0]
        d = json.loads(p.read_text(encoding="utf-8"))
        row = {"model": rk.model, "dataset": rk.dataset, "split": rk.split, "run": rk.run, "source_file": str(p)}
        row.update(d)
        collected_rows.append(row)

    # Optional compute fallback
    if missing and cfg.get("compute_if_missing", False):
        ckpt_root = Path(cfg.get("checkpoint_root", output_root / "checkpoints"))
        img_size = int(cfg.get("img_size", 640))
        device = str(cfg.get("device", "cpu"))
        warmup = int(cfg.get("warmup", 10))
        iters = int(cfg.get("iters", 50))

        for rk in missing:
            # expected checkpoint naming (adapt if needed):
            # outputs/checkpoints/<model>/<dataset>/<split>/run<k>/best.pt
            ckpt = ckpt_root / rk.model / rk.dataset / rk.split / f"run{rk.run}" / "best.pt"
            if not ckpt.exists():
                continue
            model = _load_checkpoint(ckpt)
            if model is None:
                continue

            params = count_params(model)
            lat = measure_latency_ms(model, img_size=img_size, device=device, warmup=warmup, iters=iters)

            row = {
                "model": rk.model,
                "dataset": rk.dataset,
                "split": rk.split,
                "run": rk.run,
                "params": params,
                "latency_ms": lat,
                "device": device,
                "img_size": img_size,
                "source_file": str(ckpt),
            }
            # save individual json
            out_p = out_raw / rk.model / rk.dataset / rk.split
            out_p.mkdir(parents=True, exist_ok=True)
            (out_p / f"run{rk.run}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
            collected_rows.append(row)

    if not collected_rows:
        raise FileNotFoundError(
            "No efficiency metrics found.\n"
            "Either export efficiency JSONs into outputs/metrics/raw/efficiency/..., "
            "or set compute_if_missing=true in configs/eval/efficiency.yaml AND ensure checkpoints store full nn.Module."
        )

    df = pd.DataFrame(collected_rows)
    df.to_csv(out_raw / "all_runs_efficiency.csv", index=False)
    print(f"[OK] Saved: {out_raw / 'all_runs_efficiency.csv'}")


if __name__ == "__main__":
    main()