from __future__ import annotations

from pathlib import Path
from typing import Optional, TextIO
import datetime


def make_run_log_path(output_root: Path, model: str, dataset: str, split: str, run: int) -> Path:
    p = output_root / "logs" / model / dataset / split
    p.mkdir(parents=True, exist_ok=True)
    return p / f"run{run}.log"


class TeeLogger:
    """
    Minimal logger that writes to both stdout and a log file.
    """
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.f: Optional[TextIO] = None

    def __enter__(self):
        self.f = open(self.log_path, "a", encoding="utf-8")
        self.write(f"\n===== START {datetime.datetime.now().isoformat()} =====\n")
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.f:
            self.write(f"===== END {datetime.datetime.now().isoformat()} =====\n")
            self.f.close()
            self.f = None

    def write(self, msg: str) -> None:
        print(msg, end="")
        if self.f:
            self.f.write(msg)
            self.f.flush()