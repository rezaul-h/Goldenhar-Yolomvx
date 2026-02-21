from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from ..eval.config import load_yaml
from ..eval.config import ensure_dir
from ..eval.discovery import RunKey
from ..models.baselines.base import BaselineAdapter
from ..models.baselines import build_baseline

from .seed import seed_everything
from .logging import TeeLogger, make_run_log_path
from .checkpoints import run_dir as make_run_dir
from .engine import train_torch_detector
from .config import load_train_config


def _export_metrics_json(
    *,
    output_root: Path,
    model: str,
    dataset: str,
    split: str,
    run: int,
    metrics: Dict[str, Any],
) -> Path:
    out_dir = output_root / "metrics" / "raw" / model / dataset / split
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"run{run}.json"
    p.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return p


def _try_parse_ultralytics_results(run_dir: Path) -> Optional[Dict[str, float]]:
    """
    Ultralytics typically writes:
      <run_dir>/train/results.csv
    Columns vary by version; we parse robustly.

    We export:
      precision, recall, mAP50, mAP50_95  (if present)
    """
    csv_path = run_dir / "train" / "results.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    if df.empty:
        return None

    last = df.iloc[-1].to_dict()

    # heuristic column mapping
    def pick(keys):
        for k in keys:
            if k in last:
                try:
                    return float(last[k])
                except Exception:
                    pass
        return None

    metrics = {}
    p = pick(["metrics/precision(B)", "metrics/precision", "precision"])
    r = pick(["metrics/recall(B)", "metrics/recall", "recall"])
    map50 = pick(["metrics/mAP50(B)", "metrics/mAP50", "mAP50"])
    map5095 = pick(["metrics/mAP50-95(B)", "metrics/mAP50-95", "mAP50_95", "mAP50-95"])

    if p is not None:
        metrics["precision"] = p
    if r is not None:
        metrics["recall"] = r
    if map50 is not None:
        metrics["mAP50"] = map50
    if map5095 is not None:
        metrics["mAP50_95"] = map5095

    return metrics if metrics else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a detector for a given run and export metrics artifacts.")
    ap.add_argument("--model-config", type=str, required=True, help="configs/train/<model>.yaml")
    ap.add_argument("--dataset-config", type=str, required=True, help="configs/datasets/D1.yaml or D2.yaml")
    ap.add_argument("--split-manifest", type=str, required=True, help="data/splits/.../run*_seed*.yaml")
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--output-root", type=str, default="outputs")
    args = ap.parse_args()

    cfg = load_train_config(
        model_cfg_path=args.model_config,
        dataset_cfg_path=args.dataset_config,
        split_manifest_path=args.split_manifest,
        run=args.run,
        seed=args.seed,
        output_root=args.output_root,
    )

    seed_everything(cfg.seed, deterministic=True)

    model_yaml = load_yaml(cfg.model_cfg_path)
    model_name = str(model_yaml.get("model", cfg.model_name)).lower()

    log_path = make_run_log_path(cfg.output_root, model_name, cfg.dataset, cfg.split, cfg.run)
    run_dir = make_run_dir(cfg.output_root, model_name, cfg.dataset, cfg.split, cfg.run)

    with TeeLogger(log_path) as log:
        log.write(f"[INFO] model={model_name} dataset={cfg.dataset} split={cfg.split} run={cfg.run} seed={cfg.seed}\n")
        log.write(f"[INFO] run_dir={run_dir}\n")

        # Decide execution mode: adapter (YOLO) vs torch
        baseline_obj = build_baseline(model_name, model_yaml)

        if hasattr(baseline_obj, "train") and hasattr(baseline_obj, "predict"):
            # Adapter path (Ultralytics or similar)
            adapter: BaselineAdapter = baseline_obj  # type: ignore
            log.write("[INFO] Using adapter-based training backend.\n")

            # data YAML for Ultralytics must exist; you can generate one per processed dataset.
            # We expect it in model config under: data_yaml
            data_yaml = model_yaml.get("data_yaml", None)
            if data_yaml is None:
                raise KeyError(
                    "Adapter-based training requires 'data_yaml' in the model config.\n"
                    "Example: data_yaml: data/processed/D2_yolo640/ultralytics.yaml"
                )
            data_yaml = Path(str(data_yaml))

            overrides = dict(model_yaml.get("train", {}))
            overrides["imgsz"] = cfg.img_size
            overrides["epochs"] = cfg.epochs
            overrides["batch"] = cfg.batch_size
            overrides["lr0"] = cfg.lr
            overrides["device"] = str(model_yaml.get("runtime", {}).get("device", "0"))

            best_ckpt = adapter.train(run_dir=run_dir, data_yaml=data_yaml, seed=cfg.seed, overrides=overrides)
            log.write(f"[OK] Adapter training done. best_ckpt={best_ckpt}\n")

            parsed = _try_parse_ultralytics_results(run_dir)
            if parsed is None:
                raise FileNotFoundError(
                    "Could not parse Ultralytics results.csv to export metrics.\n"
                    f"Expected: {run_dir / 'train' / 'results.csv'}"
                )
            _export_metrics_json(
                output_root=cfg.output_root,
                model=model_name,
                dataset=cfg.dataset,
                split=cfg.split,
                run=cfg.run,
                metrics=parsed,
            )
            log.write("[OK] Exported metrics JSON for eval pipeline.\n")

        else:
            # Torch training path
            log.write("[INFO] Using torch training loop.\n")

            artifacts = train_torch_detector(
                model_name=model_name,
                model_cfg=model_yaml,
                dataset_cfg_path=cfg.dataset_cfg_path,
                split_manifest_path=cfg.split_manifest_path,
                run_dir=run_dir,
                device_str=cfg.device,
                epochs=cfg.epochs,
                batch_size=cfg.batch_size,
                img_size=cfg.img_size,
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
                num_workers=cfg.num_workers,
                seed=cfg.seed,
            )
            log.write(f"[OK] Torch training done. best={artifacts.best_path} last={artifacts.last_path}\n")

            # Export minimal metrics for pipeline consistency (loss proxy)
            hist = json.loads(Path(artifacts.history_path).read_text(encoding="utf-8"))
            val_loss_last = float(hist["val_loss"][-1]) if hist.get("val_loss") else float("nan")

            metrics = {
                # for torch baselines we may not compute mAP yet; keep keys optional
                "val_loss": val_loss_last,
                "note": "For torch baselines, plug in a proper evaluator to export precision/recall/mAP.",
            }
            _export_metrics_json(
                output_root=cfg.output_root,
                model=model_name,
                dataset=cfg.dataset,
                split=cfg.split,
                run=cfg.run,
                metrics=metrics,
            )
            log.write("[OK] Exported torch metrics JSON (proxy) for pipeline.\n")


if __name__ == "__main__":
    main()