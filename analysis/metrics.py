"""Accuracy assessment metrics for satellite super-resolution validation.

Provides spatial fidelity metrics (PSNR, SSIM, NRMSE), spectral consistency
metrics (SAM, ERGAS) and an optional LPIPS-style perceptual score. All
functions accept ``(H, W)`` or ``(H, W, C)`` arrays in ``[0, 255]`` or
``[0, 1]`` and normalise internally to a ``[0, 1]`` float domain.
"""

from __future__ import annotations

import json
import numpy as np
import cv2


def _normalize(a: np.ndarray) -> np.ndarray:
    img = np.asarray(a, dtype=np.float32)
    if img.ndim == 2:
        img = img[..., np.newaxis]
    if img.max() > 1.5:
        img = img / 255.0
    return np.clip(img, 0.0, 1.0)


def psnr(hr: np.ndarray, sr: np.ndarray, data_range: float = 1.0) -> float:
    """Peak signal-to-noise ratio (dB), averaged over channels and pixels."""
    h, s = _normalize(hr).astype(np.float64), _normalize(sr).astype(np.float64)
    mse = float(np.mean((h - s) ** 2))
    if mse == 0:
        return float('inf')
    return float(10.0 * np.log10(data_range ** 2 / mse))


def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    xs = np.arange(size, dtype=np.float32) - size // 2
    kernel = np.exp(-(xs ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel[:, np.newaxis] * kernel[np.newaxis, :]


def _ssim_single(a: np.ndarray, b: np.ndarray,
                 kernel: np.ndarray, data_range: float) -> float:
    mu_a = cv2.filter2D(a, -1, kernel, borderType=cv2.BORDER_REFLECT)
    mu_b = cv2.filter2D(b, -1, kernel, borderType=cv2.BORDER_REFLECT)
    sigma_a = cv2.filter2D(a * a, -1, kernel, borderType=cv2.BORDER_REFLECT) - mu_a * mu_a
    sigma_b = cv2.filter2D(b * b, -1, kernel, borderType=cv2.BORDER_REFLECT) - mu_b * mu_b
    sigma_ab = cv2.filter2D(a * b, -1, kernel, borderType=cv2.BORDER_REFLECT) - mu_a * mu_b

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    num = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (sigma_a + sigma_b + c2)
    return float(np.mean(num / den))


def ssim(hr: np.ndarray, sr: np.ndarray, data_range: float = 1.0) -> float:
    """Structural similarity index (0-1), averaged over channels."""
    h, s = _normalize(hr).astype(np.float64), _normalize(sr).astype(np.float64)
    kernel = _gaussian_kernel()
    scores = [
        _ssim_single(h[..., c], s[..., c], kernel, data_range)
        for c in range(h.shape[2])
    ]
    return float(np.mean(scores))


def nrmse(hr: np.ndarray, sr: np.ndarray, data_range: float = 1.0) -> float:
    """Normalised root-mean-squared error (RMSE / data range)."""
    h, s = _normalize(hr).astype(np.float64), _normalize(sr).astype(np.float64)
    rmse = float(np.sqrt(np.mean((h - s) ** 2)))
    return float(rmse / data_range)


def sam(hr: np.ndarray, sr: np.ndarray) -> float:
    """Spectral angle mapper (mean degrees across pixels).

    Measures spectral distortion band-wise; 0° = identical spectral shape.
    Returns NaN if fewer than 2 bands are available.
    """
    h, s = _normalize(hr).astype(np.float64), _normalize(sr).astype(np.float64)
    c = h.shape[2]
    if c < 2:
        return float('nan')
    hp = h.reshape(-1, c)
    sp = s.reshape(-1, c)
    num = np.sum(hp * sp, axis=1)
    den = np.linalg.norm(hp, axis=1) * np.linalg.norm(sp, axis=1)
    den = np.where(den == 0, np.finfo(np.float64).eps, den)
    angles = np.arccos(np.clip(num / den, -1.0, 1.0))
    return float(np.degrees(np.mean(angles)))


def ergas(hr: np.ndarray, sr: np.ndarray, ratio: float = 4.0) -> float:
    """Erreur Relative Globale Adimensionnelle de Synthese (lower is better).

    ``ratio`` = LR pixel size / HR pixel size (i.e. the scale factor, e.g. 4
    for 10 m → 2.5 m). A value < 3 usually indicates very good fidelity.
    """
    h, s = _normalize(hr).astype(np.float64), _normalize(sr).astype(np.float64)
    c = h.shape[2]
    rmse_bands = []
    mean_bands = []
    for ch in range(c):
        rmse_bands.append(float(np.sqrt(np.mean((h[..., ch] - s[..., ch]) ** 2))))
        mean_bands.append(float(np.mean(h[..., ch])))
    rmse_bands = np.asarray(rmse_bands)
    mean_bands = np.asarray(mean_bands)
    mean_bands = np.where(mean_bands == 0, np.finfo(np.float64).eps, mean_bands)
    return float(100.0 * ratio * np.sqrt(np.mean((rmse_bands / mean_bands) ** 2)))


def _vgg_features(model, x, cut_layers):
    feats = []
    for idx, layer in enumerate(model):
        x = layer(x)
        if idx in cut_layers:
            feats.append(x)
    return feats


def _vgg_preprocess(tensor):
    import torch
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(1, 3, 1, 1)
    return (tensor - mean) / std


def lpips(hr: np.ndarray, sr: np.ndarray, device=None) -> float | None:
    """Perceptual distance (lower is better), mean over channels.

    Uses VGG19 relu features — a lightweight approximation of LPIPS.
    Returns None if torchvision is unavailable or bands < 3.
    """
    h, s = _normalize(hr).astype(np.float32), _normalize(sr).astype(np.float32)
    if h.shape[2] < 3:
        return None
    try:
        import torch
        import torch.nn.functional as F
        from torchvision.models import vgg19, VGG19_Weights
    except ImportError:
        return None

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.eval().to(device)
        for p in model.parameters():
            p.requires_grad_(False)
        cut = [1, 6, 11, 18, 25]
        x = torch.from_numpy(h[..., :3]).permute(2, 0, 1).unsqueeze(0).to(device)
        y = torch.from_numpy(s[..., :3]).permute(2, 0, 1).unsqueeze(0).to(device)
        fx, fy = _vgg_features(model, _vgg_preprocess(x), cut), \
                 _vgg_features(model, _vgg_preprocess(y), cut)
        dists = []
        for fh, fs in zip(fx, fy):
            fh, fs = F.normalize(fh, dim=1), F.normalize(fs, dim=1)
            dists.append(F.l1_loss(fh, fs).item())
        return float(np.mean(dists))
    except Exception:
        return None


def compute_metrics(hr: np.ndarray, sr: np.ndarray, scale: int = 4,
                    use_lpips: bool = True, device=None) -> dict:
    """Compute the full metric set for one HR/SR pair."""
    metrics = {
        'psnr': psnr(hr, sr),
        'ssim': ssim(hr, sr),
        'nrmse': nrmse(hr, sr),
        'sam_deg': sam(hr, sr),
        'ergas': ergas(hr, sr, ratio=scale),
    }
    if use_lpips:
        val = lpips(hr, sr, device=device)
        if val is not None:
            metrics['lpips'] = val
    return metrics


def aggregate_metrics(per_sample: list[dict]) -> dict:
    """Average a list of per-sample metric dicts, skipping NaN/inf values."""
    keys = per_sample[0].keys() if per_sample else []
    result = {}
    for key in keys:
        vals = [m[key] for m in per_sample if key in m]
        finite = [v for v in vals if v is not None and np.isfinite(v)]
        if finite:
            result[key] = float(np.mean(finite))
        elif vals:
            result[key] = None
    result['n_samples'] = len(per_sample)
    return result


def save_metrics_report(metrics: dict, path: str) -> None:
    """Write metrics to a JSON file (handles NaN/inf)."""
    import math
    clean = {}
    for k, v in metrics.items():
        if isinstance(v, float) and not math.isfinite(v):
            clean[k] = None
        else:
            clean[k] = v
    with open(path, 'w') as f:
        json.dump(clean, f, indent=2)