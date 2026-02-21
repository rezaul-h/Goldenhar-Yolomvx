from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class RunKey:
    model: str
    dataset: str
    split: str
    run: int


def _safe_int_from_runname(s: str) -> Optional[int]:
    # expects "run1" or "run01"
    s = s.strip().lower()
    if not s.startswith("run"):
        return None
    tail = s[3:]
    if not tail.isdigit():
        return None
    return int(tail)


def discover_run_logs(output_root: Path) -> List[RunKey]:
    """
    Discovers runs based on the logging structure produced by scripts:
      outputs/logs/<model>/<dataset>/<split>/run<k>.log

    This is a *stable* run registry even if different trainers are used.
    """
    logs_root = output_root / "logs"
    if not logs_root.exists():
        return []

    runs: List[RunKey] = []
    for model_dir in sorted([p for p in logs_root.iterdir() if p.is_dir()]):
        for dataset_dir in sorted([p for p in model_dir.iterdir() if p.is_dir()]):
            for split_dir in sorted([p for p in dataset_dir.iterdir() if p.is_dir()]):
                for log in sorted(split_dir.glob("run*.log")):
                    run_part = log.stem  # e.g., run1
                    run_id = _safe_int_from_runname(run_part)
                    if run_id is None:
                        continue
                    runs.append(
                        RunKey(
                            model=model_dir.name,
                            dataset=dataset_dir.name,
                            split=split_dir.name,
                            run=run_id,
                        )
                    )
    return runs


def candidate_metric_files(output_root: Path, key: RunKey) -> List[Path]:
    """
    Common patterns supported (you can add more without touching the rest):
      outputs/metrics/raw/<model>/<dataset>/<split>/run<k>.json
      outputs/metrics/raw/<model>/<dataset>/<split>/run<k>.csv
      outputs/metrics/raw/<model>__<dataset>__<split>__run<k>.json
      outputs/metrics/raw/<model>__<dataset>__<split>__run<k>.csv
    """
    raw_root = output_root / "metrics" / "raw"
    if not raw_root.exists():
        return []

    a = raw_root / key.model / key.dataset / key.split / f"run{key.run}.json"
    b = raw_root / key.model / key.dataset / key.split / f"run{key.run}.csv"
    c = raw_root / f"{key.model}__{key.dataset}__{key.split}__run{key.run}.json"
    d = raw_root / f"{key.model}__{key.dataset}__{key.split}__run{key.run}.csv"

    cand = [p for p in [a, b, c, d] if p.exists()]
    return cand


def candidate_efficiency_files(output_root: Path, key: RunKey) -> List[Path]:
    """
    Supported:
      outputs/metrics/raw/efficiency/<model>/<dataset>/<split>/run<k>.json
      outputs/metrics/raw/efficiency/<model>__<dataset>__<split>__run<k>.json
    """
    raw_root = output_root / "metrics" / "raw" / "efficiency"
    if not raw_root.exists():
        return []

    a = raw_root / key.model / key.dataset / key.split / f"run{key.run}.json"
    b = raw_root / f"{key.model}__{key.dataset}__{key.split}__run{key.run}.json"

    return [p for p in [a, b] if p.exists()]