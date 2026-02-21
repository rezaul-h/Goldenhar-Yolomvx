from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_csv_safe(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    return pd.read_csv(path)