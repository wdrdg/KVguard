"""Incremental FP32 prefix means whose persistent state does not grow with T."""

from __future__ import annotations

import torch


class RunningMean:
    """sum_t = sum_{t-1} + x_t ; mean_t = sum_t / count_t, accumulated in FP32.

    Only the new slice is cast to FP32; the persistent state is one [B, D] sum and a count.
    """

    def __init__(self, batch: int, dim: int, device: torch.device | str = "cpu"):
        self.sum = torch.zeros(batch, dim, dtype=torch.float32, device=device)
        self.count = 0

    def reset(self) -> None:
        self.sum.zero_()
        self.count = 0

    def update(self, x: torch.Tensor) -> None:
        """x: [B, n, D] new response positions (any float dtype)."""
        self.sum.add_(x.sum(dim=1, dtype=torch.float32))
        self.count += x.shape[1]

    def update_prefix_means(self, x: torch.Tensor) -> torch.Tensor:
        """Update with a chunk and return the prefix mean at every position in it: [B, n, D] FP32."""
        n = x.shape[1]
        csum = torch.cumsum(x.to(torch.float32), dim=1) + self.sum.unsqueeze(1)
        denom = torch.arange(self.count + 1, self.count + n + 1, device=x.device, dtype=torch.float32)
        means = csum / denom.view(1, n, 1)
        self.sum.copy_(csum[:, -1])
        self.count += n
        return means

    def mean(self) -> torch.Tensor:
        if self.count == 0:
            raise RuntimeError("mean of an empty running mean")
        return self.sum / self.count

    def state_bytes(self) -> int:
        return self.sum.numel() * self.sum.element_size()
