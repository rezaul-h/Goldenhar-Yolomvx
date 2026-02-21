from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .config import load_yaml
from .io import ensure_dir, read_csv_safe
from .tables import build_main_results_table, rank_models_by_metric
from .plots import plot_metric_bars, plot_confusion_matrix
from .latex import df_to_latex_booktabs, latex_results_paragraph


def _parse_cm_from_csv(path: Path) -> np.ndarray:
    """
    Accepts a raw CSV confusion matrix:
      - either with header row/col labels, or plain numeric matrix.
    """
    df = pd.read_csv(path, index_col=0)
    # if index is not numeric, still ok; use values
    return df.values.astype(float)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build publication-ready report artifacts (tables/figures/latex).")
    ap.add_argument("--config", type=str, required=True, help="configs/reports/report.yaml (you create)")
    ap.add_argument("--output-root", type=str, default="outputs", help="outputs/")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    out_root = Path(args.output_root)

    agg_path = out_root / "metrics" / "aggregated" / "metrics_mean_ci95.csv"
    agg_df = read_csv_safe(agg_path)

    report_dir = ensure_dir(out_root / "reports")
    fig_dir = ensure_dir(report_dir / "figures")
    tab_dir = ensure_dir(report_dir / "tables")
    tex_dir = ensure_dir(report_dir / "latex")

    dataset = str(cfg.get("dataset", "D2"))
    split = str(cfg.get("split", "85_15"))
    metrics = list(cfg.get("metrics", ["mAP50", "mAP50_95", "precision", "recall"]))
    key_metric = str(cfg.get("key_metric", metrics[0] if metrics else "mAP50"))
    decimals = int(cfg.get("decimals", 2))
    scale_100 = bool(cfg.get("scale_100", True))

    # ----- Main results table -----
    main_tbl = build_main_results_table(
        agg_df=agg_df,
        metrics=metrics,
        dataset=dataset,
        split=split,
        decimals=decimals,
        scale_100=scale_100,
    )
    main_csv = tab_dir / f"main_results_{dataset}_{split}.csv"
    main_tbl.to_csv(main_csv, index=False)

    main_tex = df_to_latex_booktabs(
        main_tbl,
        caption=f"Performance comparison on {dataset} ({split} split). Values are mean$\\pm$95\\% CI over repeated runs.",
        label=f"tab:main_results_{dataset}_{split}",
    )
    (tex_dir / f"main_results_{dataset}_{split}.tex").write_text(main_tex, encoding="utf-8")

    # ----- Ranking + bar plot for key metric -----
    rank = rank_models_by_metric(agg_df, metric=key_metric, dataset=dataset, split=split)
    rank_csv = tab_dir / f"ranking_{key_metric}_{dataset}_{split}.csv"
    rank.to_csv(rank_csv, index=False)

    # bar plot
    mean_col = f"{key_metric}_mean"
    ci_col = f"{key_metric}_ci95"
    sub = agg_df[(agg_df["dataset"] == dataset) & (agg_df["split"] == split)].copy()
    sub = sub.sort_values(mean_col, ascending=False)

    values = sub[mean_col].astype(float).tolist()
    errors = sub[ci_col].astype(float).tolist()
    if scale_100:
        values = [v * 100.0 for v in values]
        errors = [e * 100.0 for e in errors]

    plot_metric_bars(
        models=sub["model"].astype(str).tolist(),
        values=values,
        errors=errors,
        title=f"{key_metric} on {dataset} ({split})",
        ylabel=f"{key_metric}" + (" (%)" if scale_100 else ""),
        out_path=fig_dir / f"bar_{key_metric}_{dataset}_{split}.png",
    )

    # ----- Optional confusion matrix figure -----
    cm_cfg = cfg.get("confusion_matrix", None)
    if isinstance(cm_cfg, dict) and "path" in cm_cfg and "class_names" in cm_cfg:
        cm_path = Path(cm_cfg["path"])
        class_names = list(cm_cfg["class_names"])
        cmap = str(cm_cfg.get("cmap", "viridis"))
        normalize = cm_cfg.get("normalize", None)

        cm = _parse_cm_from_csv(cm_path)
        plot_confusion_matrix(
            cm=cm,
            class_names=class_names,
            title=str(cm_cfg.get("title", "Confusion Matrix")),
            out_path=fig_dir / f"cm_{dataset}_{split}.png",
            normalize=normalize,
            cmap=cmap,
        )

    # ----- Short results paragraph template -----
    # pick best model by key_metric_mean
    sub2 = agg_df[(agg_df["dataset"] == dataset) & (agg_df["split"] == split)].copy()
    sub2 = sub2.sort_values(mean_col, ascending=False).reset_index(drop=True)

    def _fmt(mu: float, ci: float) -> str:
        if scale_100:
            mu *= 100.0
            ci *= 100.0
        return f"{mu:.{decimals}f}$\\pm${ci:.{decimals}f}"

    best_model = str(sub2.loc[0, "model"])
    best_val = _fmt(float(sub2.loc[0, mean_col]), float(sub2.loc[0, ci_col]))
    runner_up_model = str(sub2.loc[1, "model"]) if len(sub2) > 1 else None
    runner_up_val = _fmt(float(sub2.loc[1, mean_col]), float(sub2.loc[1, ci_col])) if len(sub2) > 1 else None

    paragraph = latex_results_paragraph(
        best_model=best_model,
        dataset=dataset,
        split=split,
        key_metric=key_metric,
        best_value=best_val,
        runner_up_model=runner_up_model,
        runner_up_value=runner_up_val,
    )
    (tex_dir / f"results_paragraph_{dataset}_{split}.tex").write_text(paragraph, encoding="utf-8")

    print(f"[OK] Report saved under: {report_dir.resolve()}")
    print(f"[OK] Tables: {tab_dir.resolve()}")
    print(f"[OK] Figures: {fig_dir.resolve()}")
    print(f"[OK] LaTeX: {tex_dir.resolve()}")


if __name__ == "__main__":
    main()