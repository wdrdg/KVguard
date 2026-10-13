"""StandardScaler (fit on train only, per-response equal weight) -> Linear(D,H) -> GELU -> Linear(H,1)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

MIN_SCALE = 1e-6  # safe denominator for (near) zero-variance dimensions


def fit_weighted_scaler(x: np.ndarray, w: np.ndarray, chunk: int = 65536) -> tuple[np.ndarray, np.ndarray]:
    """Weighted mean/std in float64 accumulation. w are the per-row weights (1/n_selected),
    so every response contributes equally regardless of length."""
    w = np.asarray(w, dtype=np.float64)
    tot = w.sum()
    if tot <= 0:
        raise ValueError("scaler weights sum to zero")
    s1 = np.zeros(x.shape[1], np.float64)
    for i in range(0, len(x), chunk):
        s1 += w[i:i + chunk] @ x[i:i + chunk].astype(np.float64)
    mean = s1 / tot
    s2 = np.zeros(x.shape[1], np.float64)
    for i in range(0, len(x), chunk):
        d = x[i:i + chunk].astype(np.float64) - mean
        s2 += w[i:i + chunk] @ (d * d)
    std = np.sqrt(s2 / tot)
    scale = np.where(std < MIN_SCALE, 1.0, std)
    return mean.astype(np.float32), scale.astype(np.float32)


class ScaledMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, mean: np.ndarray | None = None, scale: np.ndarray | None = None):
        super().__init__()
        self.in_dim, self.hidden = in_dim, hidden
        self.register_buffer("mean", torch.zeros(in_dim) if mean is None else torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.ones(in_dim) if scale is None else torch.as_tensor(scale, dtype=torch.float32))
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., D] (cast to FP32) -> logits [...]."""
        x = (x.float() - self.mean) / self.scale
        return self.fc2(self.act(self.fc1(x))).squeeze(-1)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def parameter_bytes(self) -> int:
        return _unique_storage_bytes(list(self.parameters()))

    def scaler_bytes(self) -> int:
        return _unique_storage_bytes([self.mean, self.scale])


def _unique_storage_bytes(tensors) -> int:
    seen, total = set(), 0
    for t in tensors:
        st = t.untyped_storage()
        key = (st.data_ptr(), t.device)
        if key in seen:
            continue
        seen.add(key)
        total += st.nbytes()
    return total


def save_checkpoint(model: ScaledMLP, path: Path, meta: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "model.pt")
    np.savez(path / "scaler.npz", mean=model.mean.cpu().numpy(), scale=model.scale.cpu().numpy())
    (path / "meta.json").write_text(json.dumps({**meta, "in_dim": model.in_dim, "hidden": model.hidden,
                                                "n_parameters": model.n_parameters()}, indent=2))


def load_checkpoint(path: Path, device: str | torch.device = "cpu") -> tuple[ScaledMLP, dict]:
    meta = json.loads((path / "meta.json").read_text())
    m = ScaledMLP(meta["in_dim"], meta["hidden"])
    m.load_state_dict(torch.load(path / "model.pt", map_location="cpu", weights_only=True))
    return m.to(device).eval(), meta
