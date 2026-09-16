"""Uncertainty estimation for super-resolution outputs.

Super-resolution infers details that are not directly observed. To honour this
uncertainty requirement, ``estimate_uncertainty`` performs Monte-Carlo dropout:
the generator is sampled ``n_samples`` times with dropout active, and the
variance of the samples is reported as a per-pixel, per-band uncertainty map.
High variance indicates regions where the model is "guessing" texture instead
of reconstructing observed detail.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def enable_dropout(model: nn.Module, enable: bool = True) -> None:
    """Toggle only dropout layers; keeps BatchNorm/other modules in eval mode."""
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d)):
            module.train(enable)


@torch.no_grad()
def estimate_uncertainty(model: nn.Module, tensor: torch.Tensor,
                         n_samples: int = 10, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Run MC-dropout inference.

    Args:
        model: ``SuperResolutionEDSR`` built with ``dropout > 0``.
        tensor: ``(1, C, H, W)`` float tensor in ``[0, 1]``.
        n_samples: number of stochastic forward passes.

    Returns:
        ``(mean, std)`` both ``(1, C, H*s, W*s)`` in the model output domain.
        ``std`` is the per-pixel per-band standard deviation.
    """
    device = device or next(model.parameters()).device
    model.eval()
    has_dropout = any(
        isinstance(m, (nn.Dropout, nn.Dropout2d)) for m in model.modules()
    )
    if not has_dropout:
        raise RuntimeError(
            'Model has no dropout layers; uncertainty estimation unavailable. '
            'Retrain with --dropout > 0.')

    enable_dropout(model, True)
    try:
        samples = []
        for _ in range(n_samples):
            out = model(tensor.to(device))
            samples.append(out.unsqueeze(0))
        stacked = torch.cat(samples, dim=0)  # (N, 1, C, H, W)
        mean = stacked.mean(dim=0)
        std = stacked.std(dim=0)
    finally:
        enable_dropout(model, False)
    return mean, std


def aggregate_uncertainty(std_map: np.ndarray, axis: int = -1) -> np.ndarray:
    """Collapse the per-band std map into a single channel (max over bands)."""
    return np.max(std_map, axis=axis)


def summarize_uncertainty(std_map: np.ndarray, threshold: float = 0.05) -> dict:
    """Describe an uncertainty map (H, W, C) in [0, 1].

    ``threshold`` is the std above which a pixel is considered low-confidence
    (a value in the normalised image domain).
    """
    if std_map.ndim == 2:
        std_map = std_map[..., np.newaxis]
    std_map = np.asarray(std_map, dtype=np.float32)
    summary = {
        'mean_std': float(std_map.mean()),
        'p95_std': float(np.percentile(std_map, 95)),
        'max_std': float(std_map.max()),
        'threshold': float(threshold),
        'low_confidence_pct': float((std_map > threshold).mean() * 100),
    }
    summary['per_band_mean'] = {
        f'band_{i + 1}': float(std_map[..., i].mean())
        for i in range(std_map.shape[2])
    }
    return summary