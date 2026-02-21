from __future__ import annotations

from typing import Any, Dict


def flatten_dict(d: Dict[str, Any], sep: str = ".", prefix: str = "") -> Dict[str, Any]:
    """
    Flattens nested dictionaries into a single-level dict with keys joined by `sep`.
    """
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, sep=sep, prefix=key))
        else:
            out[key] = v
    return out


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default