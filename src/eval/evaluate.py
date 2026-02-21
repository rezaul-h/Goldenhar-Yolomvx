from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .config import ensure_dir, load_yaml
from .discovery import RunKey, candidate_metric_files, discover_run_logs


def _read_metrics_file(path: Path) -> Dict[str, Any]:
    """
    Accepts:
      - JSON dict (recommended)
      - CSV with one row (columns = metric keys)
    """
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if df.shape[0] < 1:
            raise ValueError(f"CSV metrics file has no rows: {path}")
        row = df.iloc[0].to_dict()
        return row
    raise ValueError(f"Unsupported metrics file type: {path}")


def _select_best_candidate(files: List[Path]) -> Path:
    # If multiple exist, prefer JSON over CSV
    files = sorted(files)
    for p in files:
        if p.suffix.lower() == ".json":
            return p
    return files[0]


def build_run_metrics_table(output_root: Path, runs: List[RunKey]) -> pd.DataFrame:
    """
    Builds a per-run table by reading existing metric artifacts.

    Expected minimal keys (you can add more):
      - mAP50
      - mAP50_95
      - precision
      - recall
    """
    rows: List[Dict[str, Any]] = []

    missing: List[RunKey] = []
    for rk in runs:
        cand = candidate_metric_files(output_root, rk)
        if not cand:
            missing.append(rk)
            continue
        f = _select_best_candidate(cand)
        d = _read_metrics_file(f)

        row = {
            "model": rk.model,
            "dataset": rk.dataset,
            "split": rk.split,
            "run": rk.run,
            "source_file": str(f),
        }
        # keep all metric keys present
        for k, v in d.items():
            row[k] = v
        rows.append(row)

    if missing:
        # Fail fast but explain how to fix
        example = missing[0]
        raise FileNotFoundError(
            "No per-run metric artifacts found for some runs.\n"
            "Expected files like:\n"
            f"  outputs/metrics/raw/{example.model}/{example.dataset}/{example.split}/run{example.run}.json\n"
            "or\n"
            f"  outputs/metrics/raw/{example.model}__{example.dataset}__{example.split}__run{example.run}.json\n"
            "\n"
            "Fix: export evaluation metrics from your trainer/evaluator into outputs/metrics/raw/ using one of the patterns above."
        )

    df = pd.DataFrame(rows)

    # normalize numeric columns where possible
    for c in df.columns:
        if c in {"model", "dataset", "split", "source_file"}:
            continue
        if c == "run":
            continue
        df[c] = pd.to_numeric(df[c], errors="ignore")

    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect per-run metrics into a unified table.")
    ap.add_argument("--config", type=str, required=True, help="configs/eval/metrics.yaml")
    ap.add_argument("--output-root", type=str, required=True, help="outputs/")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    output_root = Path(args.output_root)

    out_raw = ensure_dir(output_root / "metrics" / "raw")
    out_agg = ensure_dir(output_root / "metrics" / "aggregated")

    # Discover runs based on logs
    runs = discover_run_logs(output_root)
    if len(runs) == 0:
        raise FileNotFoundError(
            f"No runs discovered under {output_root / 'logs'}.\n"
            "Run the training scripts first (they create outputs/logs/...)."
        )

    df = build_run_metrics_table(output_root, runs)

    # Save per-run table
    per_run_path = out_raw / "all_runs_metrics.csv"
    df.to_csv(per_run_path, index=False)

    # Also create a compact view if configured
    key_metrics = cfg.get("metrics", ["mAP50", "mAP50_95", "precision", "recall"])
    cols = ["model", "dataset", "split", "run"] + [m for m in key_metrics if m in df.columns]
    compact = df[cols].copy()
    compact_path = out_raw / "all_runs_metrics_compact.csv"
    compact.to_csv(compact_path, index=False)

    print(f"[OK] Saved: {per_run_path}")
    print(f"[OK] Saved: {compact_path}")


if __name__ == "__main__":
    main()