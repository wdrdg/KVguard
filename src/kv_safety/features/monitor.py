"""Online streaming detector with one incremental implementation per method.

Lifecycle per request:
    det.begin_request()            # resets all per-request statistics
    model(prompt)                  # prefill: not counted
    det.start_response()
    for each response token t:     # feeding token t writes its K/V and h_t
        out = model(token_t, past_key_values=cache)
        score_t = det.after_forward(cache, n_new=1)   # score of the prefix that ends at t
The score returned after feeding token t belongs to prefix t (1-based), never t+1.
"""

from __future__ import annotations

import torch

from ..classifier import ScaledMLP
from .cache_reader import check_cache, new_kv
from .hidden_hook import BlockOutputHook
from .running_mean import RunningMean

METHODS = ("baseline_no_detector", "h_last", "h_mean", "k_mean", "v_mean", "kv_mean")
FEATURE_DIMS = {"h_last": "hidden", "h_mean": "hidden", "k_mean": "kv", "v_mean": "kv", "kv_mean": "kv2"}


def feature_dim(method: str, hidden_size: int, kv_dim: int) -> int:
    return {"hidden": hidden_size, "kv": kv_dim, "kv2": 2 * kv_dim}[FEATURE_DIMS[method]]


class StreamingDetector:
    def __init__(self, model, method: str, layer_idx: int, classifier: ScaledMLP | None,
                 batch: int = 1, device: torch.device | str | None = None):
        if method not in METHODS:
            raise ValueError(method)
        cfg = model.config
        self.method, self.layer_idx, self.batch = method, layer_idx, batch
        self.device = device or next(model.parameters()).device
        self.classifier = classifier
        self.counting = False
        self.hook = None
        hidden = cfg.hidden_size
        kv_dim = cfg.num_key_value_heads * getattr(cfg, "head_dim", hidden // cfg.num_attention_heads)
        self.state: dict[str, torch.Tensor] = {}
        self.means: dict[str, RunningMean] = {}
        if method == "h_last":
            self.state["last"] = torch.zeros(batch, hidden, dtype=torch.float32, device=self.device)
            self.n_seen = 0
        elif method == "h_mean":
            self.means["h"] = RunningMean(batch, hidden, self.device)
        elif method in ("k_mean", "kv_mean"):
            self.means["k"] = RunningMean(batch, kv_dim, self.device)
        if method in ("v_mean", "kv_mean"):
            self.means["v"] = RunningMean(batch, kv_dim, self.device)
        if method in ("h_last", "h_mean"):
            self.hook = BlockOutputHook(model, layer_idx, self._on_block_output)
        if method != "baseline_no_detector" and classifier is not None:
            exp = feature_dim(method, hidden, kv_dim)
            if classifier.in_dim != exp:
                raise ValueError(f"classifier in_dim {classifier.in_dim} != feature dim {exp} for {method}")

    # ---- lifecycle -------------------------------------------------------------------------
    def begin_request(self) -> None:
        self.counting = False
        for m in self.means.values():
            m.reset()
        if "last" in self.state:
            self.state["last"].zero_()
            self.n_seen = 0

    def start_response(self) -> None:
        self.counting = True

    def end_request(self) -> None:
        self.counting = False

    def remove(self) -> None:
        if self.hook is not None:
            self.hook.remove()

    # ---- updates ---------------------------------------------------------------------------
    def _on_block_output(self, h: torch.Tensor) -> None:
        if not self.counting:
            return
        if self.method == "h_mean":
            self.means["h"].update(h)
        else:
            self.state["last"].copy_(h[:, -1])
            self.n_seen += h.shape[1]

    def current_feature(self) -> torch.Tensor | None:
        m = self.method
        if m == "baseline_no_detector":
            return None
        if m == "h_last":
            return self.state["last"]
        if m == "h_mean":
            return self.means["h"].mean()
        if m == "k_mean":
            return self.means["k"].mean()
        if m == "v_mean":
            return self.means["v"].mean()
        return torch.cat([self.means["k"].mean(), self.means["v"].mean()], dim=-1)

    @torch.inference_mode()
    def after_forward(self, cache, n_new: int = 1) -> torch.Tensor | None:
        """Call after each response forward. Returns logits [B] for the prefix now in the cache."""
        if not self.counting or self.method == "baseline_no_detector":
            return None
        if self.method in ("k_mean", "v_mean", "kv_mean"):
            k, v = new_kv(cache, self.layer_idx, n_new)
            if "k" in self.means:
                self.means["k"].update(k)
            if "v" in self.means:
                self.means["v"].update(v)
        feat = self.current_feature()
        if self.classifier is None:
            return None
        return self.classifier(feat)

    # ---- audit -------------------------------------------------------------------------------
    def persistent_state_bytes(self) -> int:
        tensors = list(self.state.values()) + [m.sum for m in self.means.values()]
        seen, total = set(), 0
        for t in tensors:
            st = t.untyped_storage()
            if st.data_ptr() in seen:
                continue
            seen.add(st.data_ptr())
            total += st.nbytes()
        return total


def check_cache_for_monitor(cache) -> None:
    check_cache(cache)
