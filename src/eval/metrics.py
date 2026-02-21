from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class SummaryStats:
    mean: float
    std: float
    ci95: float
    n: int


def mean_std_ci95(x: List[float]) -> SummaryStats:
    """
    95% CI using normal approximation: 1.96 * std/sqrt(n).
    With n=4 runs this is commonly used in ML papers; report n explicitly.
    """
    arr = np.asarray(x, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = int(arr.size)
    if n == 0:
        return SummaryStats(mean=float("nan"), std=float("nan"), ci95=float("nan"), n=0)
    m = float(arr.mean())
    s = float(arr.std(ddof=1)) if n > 1 else 0.0
    ci = float(1.96 * s / np.sqrt(n)) if n > 1 else 0.0
    return SummaryStats(mean=m, std=s, ci95=ci, n=n)


def paired_tests(a: List[float], b: List[float]) -> Dict[str, float]:
    """
    Paired t-test and Wilcoxon signed-rank test (if SciPy is available).
    Returns p-values; if SciPy not installed, returns NaN for tests.
    """
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)

    # align + drop NaNs
    mask = ~np.isnan(aa) & ~np.isnan(bb)
    aa = aa[mask]
    bb = bb[mask]

    out = {"p_ttest": float("nan"), "p_wilcoxon": float("nan"), "n": float(aa.size)}
    if aa.size < 2:
        return out

    try:
        from scipy.stats import ttest_rel, wilcoxon  # type: ignore
        out["p_ttest"] = float(ttest_rel(aa, bb).pvalue)
        # Wilcoxon may fail if all differences are zero
        try:
            out["p_wilcoxon"] = float(wilcoxon(aa, bb, zero_method="wilcox").pvalue)
        except Exception:
            out["p_wilcoxon"] = float("nan")
    except Exception:
        # SciPy not available
        pass

    return out