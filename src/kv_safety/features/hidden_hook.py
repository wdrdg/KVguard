"""Forward hook on one decoder block that hands its output (residual stream, pre final norm)
to a callback and keeps no reference afterwards."""

from __future__ import annotations

from typing import Callable

import torch


def block_output(out) -> torch.Tensor:
    return out[0] if isinstance(out, (tuple, list)) else out


class BlockOutputHook:
    def __init__(self, model, layer_idx: int, callback: Callable[[torch.Tensor], None]):
        self.layer = get_decoder_layers(model)[layer_idx]
        self.callback = callback
        self.enabled = True
        self.handle = self.layer.register_forward_hook(self._hook)

    def _hook(self, module, args, out):
        if self.enabled:
            self.callback(block_output(out))
        return None  # never modifies the output

    def remove(self) -> None:
        self.handle.remove()


def get_decoder_layers(model):
    base = model.model if hasattr(model, "model") and hasattr(model.model, "layers") else model
    return base.layers
