from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional


def project_root(marker: str = "pyproject.toml") -> Path:
    """
    Attempts to locate project root by searching upwards for a marker file.
    Falls back to current working directory.
    """
    p = Path.cwd().resolve()
    for parent in [p] + list(p.parents):
        if (parent / marker).exists():
            return parent
    return p


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_files(
    root: str | Path,
    exts: Optional[Iterable[str]] = None,
    recursive: bool = True,
) -> List[Path]:
    root = Path(root)
    if not root.exists():
        return []
    if exts is not None:
        exts = {e.lower().lstrip(".") for e in exts}

    files = []
    it = root.rglob("*") if recursive else root.glob("*")
    for f in it:
        if not f.is_file():
            continue
        if exts is None:
            files.append(f)
        else:
            if f.suffix.lower().lstrip(".") in exts:
                files.append(f)
    return sorted(files)