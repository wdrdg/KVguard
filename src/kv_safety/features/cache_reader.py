"""Read the native (post k-norm, post-RoPE) K/V written to a DynamicCache for one layer."""

from __future__ import annotations

import torch


def check_model_cache_compat(config) -> None:
    """The first version supports only full-attention, unquantised, non-evicting caches."""
    if getattr(config, "use_sliding_window", False):
        raise ValueError("sliding-window attention is not supported by the running-mean reader")
    lt = getattr(config, "layer_types", None)
    if lt and any(t != "full_attention" for t in lt):
        raise ValueError(f"non full-attention layers present: {set(lt)}")


def check_cache(cache) -> None:
    from transformers import DynamicCache

    if type(cache) is not DynamicCache:
        raise TypeError(f"only DynamicCache is supported, got {type(cache).__name__}")
    if hasattr(cache, "layers"):
        from transformers.cache_utils import DynamicLayer

        bad = {type(l).__name__ for l in cache.layers if type(l) is not DynamicLayer}
        if bad:
            raise TypeError(f"unsupported cache layer types {bad} (sliding/quantised/offloaded)")


def layer_kv(cache, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """[B, H_kv, L, d] key/value tensors stored by the model (views, not copies)."""
    if hasattr(cache, "layers"):
        layer = cache.layers[layer_idx]
        return layer.keys, layer.values
    return cache.key_cache[layer_idx], cache.value_cache[layer_idx]  # transformers < 4.56


def flatten_heads(x: torch.Tensor) -> torch.Tensor:
    """[B, H_kv, n, d] -> [B, n, H_kv*d], head-major (head 0 dims 0..d-1, then head 1, ...)."""
    b, h, n, d = x.shape
    return x.permute(0, 2, 1, 3).reshape(b, n, h * d)


def new_kv(cache, layer_idx: int, n_new: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The K/V of the last n_new positions, flattened per token: two [B, n_new, H_kv*d] views/tensors."""
    k, v = layer_kv(cache, layer_idx)
    return flatten_heads(k[:, :, -n_new:, :]), flatten_heads(v[:, :, -n_new:, :])
