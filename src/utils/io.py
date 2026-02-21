from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def read_text(path: str | Path, encoding: str = "utf-8") -> str:
    p = Path(path)
    return p.read_text(encoding=encoding)


def write_text(path: str | Path, text: str, encoding: str = "utf-8") -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding=encoding)
    return p


def read_json(path: str | Path) -> Any:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(path: str | Path, obj: Any, indent: int = 2) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=indent), encoding="utf-8")
    return p


def read_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping/dict: {p}")
    return data


def write_yaml(path: str | Path, obj: Dict[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)
    return p