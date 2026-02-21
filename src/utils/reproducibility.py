from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

try:
    import torch
except Exception:
    torch = None  # type: ignore


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def set_determinism(enable: bool = True) -> None:
    """
    Toggle deterministic behavior for torch/cudnn.
    """
    if torch is None:
        return
    torch.backends.cudnn.deterministic = bool(enable)
    torch.backends.cudnn.benchmark = not bool(enable)