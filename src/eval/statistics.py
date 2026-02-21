from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import ensure_dir, load_yaml
from .metrics import mean_std_ci95, paired_tests


def _load_runs_table(output_root: Path) -> pd.DataFrame:
    p = output_root / "metrics" / "raw" / "all_runs_metrics.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"Missing per-run metrics table: {p}\n"
            "Run: python -m src.eval.evaluate --config <...> --output-root outputs"
        )
    df = pd.read_csv(p)
    return df


def aggregate_by_group(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    group_cols = ["model", "dataset", "split"]

    for (model, dataset, split), g in df.groupby(group_cols):
        row = {"model": model, "dataset": dataset, "split": split, "n_runs": int(g["run"].nunique())}
        for m in metrics:
            if m not in g.columns:
                continue
            vals = pd.to_numeric(g[m], errors="coerce").dropna().tolist()
            s = mean_std_ci95(vals)
            row[f"{m}_mean"] = s.mean
            row[f"{m}_std"] = s.std
            row[f"{m}_ci95"] = s.ci95
            row[f"{m}_n"] = s.n
        rows.append(row)

    out = pd.DataFrame(rows).sort_values(["dataset", "split", "model"])
    return out


def paired_significance(df: pd.DataFrame, metric: str, baseline: str) -> pd.DataFrame:
    """
    For each dataset+split, compare each model vs baseline using paired tests across runs.
    Assumes runs are aligned by run id (1..4).
    """
    rows: List[Dict[str, Any]] = []
    for (dataset, split), g in df.groupby(["dataset", "split"]):
        gb = g[g["model"] == baseline]
        if gb.empty or metric not in gb.columns:
            continue
        # run->value for baseline
        b_map = {int(r): float(v) for r, v in zip(gb["run"], pd.to_numeric(gb[metric], errors="coerce"))}

        for model in sorted(g["model"].unique()):
            if model == baseline:
                continue
            gm = g[g["model"] == model]
            if gm.empty or metric not in gm.columns:
                continue
            m_map = {int(r): float(v) for r, v in zip(gm["run"], pd.to_numeric(gm[metric], errors="coerce"))}

            # align runs
            common_runs = sorted(set(b_map.keys()) & set(m_map.keys()))
            a = [m_map[r] for r in common_runs]
            b = [b_map[r] for r in common_runs]

            tests = paired_tests(a=a, b=b)  # model vs baseline
            rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "metric": metric,
                    "baseline": baseline,
                    "model": model,
                    "n_pairs": int(tests.get("n", 0)),
                    "p_ttest": tests.get("p_ttest", float("nan")),
                    "p_wilcoxon": tests.get("p_wilcoxon", float("nan")),
                }
            )

    return pd.DataFrame(rows).sort_values(["dataset", "split", "metric", "model"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate runs + compute CIs and significance tests.")
    ap.add_argument("--config", type=str, required=True, help="configs/eval/stats.yaml")
    ap.add_argument("--output-root", type=str, required=True, help="outputs/")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    output_root = Path(args.output_root)

    out_agg = ensure_dir(output_root / "metrics" / "aggregated")

    df = _load_runs_table(output_root)

    metrics = cfg.get("metrics", ["mAP50", "mAP50_95", "precision", "recall"])
    baseline = str(cfg.get("baseline_model", "yolov9"))

    # Aggregate (mean/std/CI)
    agg = aggregate_by_group(df, metrics=metrics)
    agg_path = out_agg / "metrics_mean_ci95.csv"
    agg.to_csv(agg_path, index=False)

    # Significance vs baseline (paired)
    sig_tables = []
    for m in metrics:
        if m in df.columns:
            sig_tables.append(paired_significance(df, metric=m, baseline=baseline))
    sig = pd.concat(sig_tables, axis=0, ignore_index=True) if sig_tables else pd.DataFrame()
    sig_path = out_agg / "significance_vs_baseline.csv"
    sig.to_csv(sig_path, index=False)

    print(f"[OK] Saved: {agg_path}")
    print(f"[OK] Saved: {sig_path}")


if __name__ == "__main__":
    main()