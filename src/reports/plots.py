from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt


def _setup_matplotlib():
    # Journal-friendly defaults (no explicit colors; rely on matplotlib defaults)
    plt.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })


def plot_metric_bars(
    models: List[str],
    values: List[float],
    errors: Optional[List[float]],
    title: str,
    ylabel: str,
    out_path: str | Path,
) -> Path:
    """
    Simple bar plot with error bars (CI95).
    """
    _setup_matplotlib()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(models))
    fig = plt.figure(figsize=(max(6.0, 0.75 * len(models)), 4.2))
    ax = fig.add_subplot(111)

    if errors is None:
        ax.bar(x, values)
    else:
        ax.bar(x, values, yerr=errors, capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_learning_curves(
    epochs: Sequence[int],
    curves: Dict[str, Dict[str, Sequence[float]]],
    title: str,
    ylabel: str,
    out_path: str | Path,
) -> Path:
    """
    curves example:
      {
        "train": {"run1": [...], "run2": [...], ...},
        "val":   {"run1": [...], ...}
      }
    Plots all runs with distinct labels (line styles are default).
    """
    _setup_matplotlib()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7.0, 4.3))
    ax = fig.add_subplot(111)

    for split_name, runs in curves.items():
        for run_name, y in runs.items():
            ax.plot(list(epochs), list(y), label=f"{split_name}-{run_name}")

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", ncol=2, frameon=True)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str,
    out_path: str | Path,
    normalize: Optional[str] = None,  # None | "true" | "pred"
    cmap: str = "viridis",
) -> Path:
    """
    Q1-friendly confusion matrix:
      - optional normalization
      - annotated cells (adaptive text color)
      - clean axis labels
    """
    _setup_matplotlib()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cm = np.asarray(cm, dtype=float)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("cm must be square (C,C).")
    C = cm.shape[0]
    if len(class_names) != C:
        raise ValueError("class_names length must match cm size.")

    disp = cm.copy()
    if normalize == "true":
        disp = disp / (disp.sum(axis=1, keepdims=True) + 1e-12)
    elif normalize == "pred":
        disp = disp / (disp.sum(axis=0, keepdims=True) + 1e-12)

    fig = plt.figure(figsize=(7.2, 6.3))
    ax = fig.add_subplot(111)
    im = ax.imshow(disp, interpolation="nearest", cmap=cmap)

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks(np.arange(C))
    ax.set_yticks(np.arange(C))
    ax.set_xticklabels(class_names, rotation=35, ha="right")
    ax.set_yticklabels(class_names)

    # colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Proportion" if normalize else "Count", rotation=90)

    # annotations
    thresh = disp.max() * 0.55
    for i in range(C):
        for j in range(C):
            val = disp[i, j]
            if normalize:
                txt = f"{val:.2f}"
            else:
                txt = f"{int(cm[i, j])}"
            ax.text(
                j, i, txt,
                ha="center", va="center",
                color="white" if val > thresh else "black",
                fontsize=10
            )

    ax.grid(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path