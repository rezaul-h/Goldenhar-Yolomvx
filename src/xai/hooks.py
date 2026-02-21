from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class HookOutput:
    activation: Optional[torch.Tensor] = None
    gradient: Optional[torch.Tensor] = None


class ActivationGradientHook:
    """
    Captures forward activations and backward gradients for a specific module.
    Suitable for Grad-CAM style attribution.
    """
    def __init__(self, module: nn.Module):
        self.module = module
        self.out = HookOutput()
        self._fwd = None
        self._bwd = None

    def _forward_hook(self, m: nn.Module, inp: Tuple[Any, ...], out: Any):
        if isinstance(out, torch.Tensor):
            self.out.activation = out

    def _backward_hook(self, m: nn.Module, grad_in: Tuple[Any, ...], grad_out: Tuple[Any, ...]):
        # grad_out[0] corresponds to gradient of output of the hooked module
        if grad_out and isinstance(grad_out[0], torch.Tensor):
            self.out.gradient = grad_out[0]

    def attach(self) -> "ActivationGradientHook":
        self._fwd = self.module.register_forward_hook(self._forward_hook)
        # full backward hook is safer for modern torch
        self._bwd = self.module.register_full_backward_hook(self._backward_hook)
        return self

    def remove(self) -> None:
        if self._fwd is not None:
            self._fwd.remove()
            self._fwd = None
        if self._bwd is not None:
            self._bwd.remove()
            self._bwd = None

    def clear(self) -> None:
        self.out = HookOutput()