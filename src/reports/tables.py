from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _fmt_mean_ci(mean: float, ci: float, decimals: int = 2, scale_100: bool = True) -> str:
    """
    Default journal-friendly formatting: mean ± CI.
    If metrics are in [0,1], set scale_100=True to show percentages.
    """
    if np.isnan(mean) or np.isnan(ci):
        return "NA"
    if scale_100:
        mean *= 100.0
        ci *= 100.0
    return f"{mean:.{decimals}f}$\\pm${ci:.{decimals}f}"


def rank_models_by_metric(
    agg_df: pd.DataFrame,
    metric: str,
    dataset: str,
    split: str,
    higher_is_better: bool = True,
) -> pd.DataFrame:
    """
    Returns a sorted ranking table for a single dataset+split.
    Expects columns: <metric>_mean and <metric>_ci95.
    """
    m_mean = f"{metric}_mean"
    m_ci = f"{metric}_ci95"
    required = {"model", "dataset", "split", m_mean, m_ci}
    missing = [c for c in required if c not in agg_df.columns]
    if missing:
        raise KeyError(f"Missing columns in aggregated df: {missing}")

    sub = agg_df[(agg_df["dataset"] == dataset) & (agg_df["split"] == split)].copy()
    sub = sub.sort_values(m_mean, ascending=not higher_is_better).reset_index(drop=True)
    sub["rank"] = np.arange(1, len(sub) + 1)
    return sub[["rank", "model", m_mean, m_ci, f"{metric}_n"] if f"{metric}_n" in sub.columns else ["rank", "model", m_mean, m_ci]]


def build_main_results_table(
    agg_df: pd.DataFrame,
    metrics: List[str],
    dataset: str,
    split: str,
    decimals: int = 2,
    scale_100: bool = True,
) -> pd.DataFrame:
    """
    Builds a compact table:
      rows = models
      cols = metrics formatted as mean ± CI95

    Input: aggregated CSV from `outputs/metrics/aggregated/metrics_mean_ci95.csv`
    """
    sub = agg_df[(agg_df["dataset"] == dataset) & (agg_df["split"] == split)].copy()
    if sub.empty:
        raise ValueError(f"No rows for dataset={dataset}, split={split}.")

    out = pd.DataFrame({"Model": sub["model"].astype(str).tolist()})

    for m in metrics:
        mean_col = f"{m}_mean"
        ci_col = f"{m}_ci95"
        if mean_col not in sub.columns or ci_col not in sub.columns:
            continue
        out[m] = [
            _fmt_mean_ci(float(mu), float(ci), decimals=decimals, scale_100=scale_100)
            for mu, ci in zip(sub[mean_col].values, sub[ci_col].values)
        ]

    # Optional: sort by the first metric
    if metrics:
        sort_key = metrics[0]
        if sort_key in out.columns:
            # parse mean from the formatted string if possible; fallback to original ordering
            def _parse_mean(s: str) -> float:
                try:
                    return float(s.split("$\\pm$")[0])
                except Exception:
                    return float("nan")
            out["_sort"] = out[sort_key].map(_parse_mean)
            out = out.sort_values("_sort", ascending=False).drop(columns=["_sort"]).reset_index(drop=True)

    return out


def build_per_class_table_from_confusion(
    cm: np.ndarray,
    class_names: List[str],
    decimals: int = 2,
    scale_100: bool = True,
) -> pd.DataFrame:
    """
    Build per-class precision/recall/F1 from a confusion matrix.

    cm: shape (C,C), rows=true, cols=pred.
    """
    cm = np.asarray(cm, dtype=float)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("Confusion matrix must be square (C,C).")
    C = cm.shape[0]
    if len(class_names) != C:
        raise ValueError("len(class_names) must match cm size.")

    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    prec = np.divide(tp, tp + fp + 1e-12)
    rec = np.divide(tp, tp + fn + 1e-12)
    f1 = np.divide(2 * prec * rec, prec + rec + 1e-12)
    support = cm.sum(axis=1)

    def _fmt(x: float) -> float:
        return float(x * 100.0) if scale_100 else float(x)

    df = pd.DataFrame(
        {
            "Class": class_names,
            "Precision": [round(_fmt(x), decimals) for x in prec],
            "Recall": [round(_fmt(x), decimals) for x in rec],
            "F1": [round(_fmt(x), decimals) for x in f1],
            "Support": support.astype(int),
        }
    )
    return df