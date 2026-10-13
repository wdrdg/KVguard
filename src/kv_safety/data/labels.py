"""Cumulative prefix-harm proxy labels and training position selection.

All positions are 1-based response-token positions. For the first harmful sentence
covering response tokens [a, b] (inclusive):
    t <  a      -> 0
    a <= t < b  -> IGNORE (-100)
    t >= b      -> 1
A fully safe response is 0 everywhere.
"""

from __future__ import annotations

import numpy as np

IGNORE = -100


def prefix_labels(T: int, a: int | None, b: int | None) -> np.ndarray:
    if T <= 0:
        raise ValueError("empty response")
    lab = np.zeros(T, dtype=np.int64)
    if a is None:
        return lab
    if not (1 <= a <= b <= T):
        raise ValueError(f"invalid harmful span a={a} b={b} T={T}")
    t = np.arange(1, T + 1)
    lab[(t >= a) & (t < b)] = IGNORE
    lab[t >= b] = 1
    return lab


def select_train_positions(labels: np.ndarray, a: int | None, b: int | None,
                           max_n: int = 32, after: int = 32) -> np.ndarray:
    """Up to max_n valid (non-ignored) positions: always a-1, b, b+after when valid, the rest
    spread evenly over the remaining valid positions. Returns sorted unique 1-based positions."""
    T = len(labels)
    valid = np.flatnonzero(labels != IGNORE) + 1
    if len(valid) <= max_n:
        return valid
    valid_set = set(valid.tolist())
    prio = []
    if a is not None:
        for p in (a - 1, b, b + after):
            if 1 <= p <= T and p in valid_set and p not in prio:
                prio.append(p)
    prio = prio[:max_n]
    rest = np.array([p for p in valid if p not in set(prio)])
    k = max_n - len(prio)
    if k > 0 and len(rest):
        idx = np.unique(np.round(np.linspace(0, len(rest) - 1, k)).astype(int))
        chosen = rest[idx]
    else:
        chosen = np.array([], dtype=int)
    out = np.unique(np.concatenate([np.array(prio, dtype=int), chosen.astype(int)]))
    assert len(out) <= max_n and out.min() >= 1 and out.max() <= T
    return out


def position_weights(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n, dtype=np.float32)
