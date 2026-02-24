from __future__ import annotations

import torch


def tensor_stats(tensor: torch.Tensor, bins: int = 20) -> dict:
    """Compute summary statistics for a weight tensor."""
    t = tensor.float().detach()
    std = t.std().item()
    mean = t.mean().item()

    return {
        "mean": mean,
        "std": std,
        "min_val": t.min().item(),
        "max_val": t.max().item(),
        "l2_norm": t.norm().item(),
        "shape": list(t.shape),
        "num_outliers": int((t.abs() > abs(mean) + 3 * std).sum().item()),
        "histogram": torch.histc(t.cpu(), bins=bins).tolist(),
    }
