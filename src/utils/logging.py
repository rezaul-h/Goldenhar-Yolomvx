from __future__ import annotations

import datetime
from typing import Optional


def get_time_str() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_line(msg: str, prefix: Optional[str] = None) -> str:
    """
    Returns a formatted log line; caller may print or write to file.
    """
    t = get_time_str()
    if prefix:
        return f"[{t}] [{prefix}] {msg}"
    return f"[{t}] {msg}"