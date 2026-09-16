#!/usr/bin/env python3
"""Accuracy assessment against high-resolution references.

Runs the trained super-resolution model over the held-out test split and
reports spatial fidelity (PSNR / SSIM / NRMSE), spectral consistency
(SAM / ERGAS) and an optional perceptual score (LPIPS-style). Outputs:

- ``<out>/sr_metrics.json`` — averaged + per-sample metrics
- ``<out>/montage_*.png``    — LR / super-resolved / reference comparison tiles

Usage:

    python -m training.validate --checkpoint checkpoints/satelite_sr_best.pt \
        --manifest satellite_sr_manifest.csv --out evaluation/
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analysis.metrics import compute_metrics, aggregate_metrics, save_metrics_report
from models.enhance import SatelliteSuperResolution
from training.dataset import build_sr_datasets


def _to_uint8_rgb(img: np.ndarray) -> np.ndarray:
    """Convert (H,W,C) float [0,1] to uint8 RGB for visualisation."""
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    if img.shape[2] == 1:
        return cv2.cvtColor(img[..., 0], cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 2:
        return cv2.merge([img[..., 0], img[..., 0], img[..., 1]])
    return img[..., :3]


def make_montage(lr: np.ndarray, sr: np.ndarray, hr: np.ndarray, scale: int) -> np.ndarray:
    lr_up = cv2.resize(lr, (hr.shape[1], hr.shape[0]), interpolation=cv2.INTER_NEAREST)
    tiles = [_to_uint8_rgb(t) for t in (lr_up, sr, hr)]
    height = max(t.shape[0] for t in tiles)
    width = max(t.shape[1] for t in tiles)
    tiles = [
        cv2.resize(t, (width, height), interpolation=cv2.INTER_NEAREST)
        for t in tiles
    ]
    label_h = 28
    bar = np.full((height + label_h, 4, 3), 255, dtype=np.uint8)
    panel = np.full((height + label_h, width, 3), 255, dtype=np.uint8)
    cols = []
    for title, tile in zip(['LR (input)', 'Super-resolved', 'Reference (HR)'], tiles):
        panel[:] = 255
        panel[:height] = tile
        cv2.putText(panel, title, (8, height + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cols.append(panel)
        cols.append(bar.copy())
    return np.hstack(cols[:-1])


def main():
    p = argparse.ArgumentParser(description='Super-resolution validation')
    p.add_argument('--checkpoint', type=str,
                   default='checkpoints/satelite_sr.pt')
    p.add_argument('--manifest', type=str, default=None)
    p.add_argument('--images-dir', type=str, default=None)
    p.add_argument('--root', type=str, default=None)
    p.add_argument('--scale', type=int, default=4)
    p.add_argument('--patch', type=int, default=256)
    p.add_argument('--out', type=str, default='evaluation')
    p.add_argument('--no-lpips', action='store_true',
                   help='Skip the VGG perceptual metric')
    p.add_argument('--limit-test', type=int, default=None)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, _, test_ds, _ = build_sr_datasets(
        manifest=args.manifest, images_dir=args.images_dir, root=args.root,
        scale=args.scale, hr_patch_size=args.patch,
    )
    if args.limit_test:
        test_ds.samples = test_ds.samples[:args.limit_test]
    print(f'Test samples: {len(test_ds)}')

    enhancer = SatelliteSuperResolution(
        scale=args.scale,
        n_channels=test_ds.n_channels,
        checkpoint=args.checkpoint,
        device=device,
    )
    if enhancer.model is None:
        raise RuntimeError(f'Could not load checkpoint: {args.checkpoint}')

    loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    per_sample = []
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, (lr, hr) in enumerate(loader):
        lr_np = lr.squeeze(0).permute(1, 2, 0).numpy()  # (H, W, C) float [0,1]
        hr_np = hr.squeeze(0).permute(1, 2, 0).numpy()
        with torch.no_grad():
            out = enhancer.model(lr.to(device))
        sr_np = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
        sr_np = np.clip(sr_np, 0, 1).astype(np.float32)

        metrics = compute_metrics(hr_np, sr_np, scale=args.scale,
                                  use_lpips=not args.no_lpips, device=device)
        per_sample.append(metrics)
        print(f'[{i+1}/{len(test_ds)}] PSNR {metrics["psnr"]:.2f} dB  '
              f'SSIM {metrics["ssim"]:.4f}  SAM {metrics["sam_deg"]:.2f} deg  '
              f'ERGAS {metrics["ergas"]:.2f}')

        if i < 6:
            montage = make_montage(lr_np, sr_np, hr_np, args.scale)
            cv2.imwrite(str(out_dir / f'montage_{i:02d}.png'),
                        cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))

    report = aggregate_metrics(per_sample)
    save_metrics_report(report, str(out_dir / 'sr_metrics.json'))
    print('\n=== AGGREGATE TEST METRICS ===')
    for k in ['psnr', 'ssim', 'nrmse', 'sam_deg', 'ergas', 'lpips']:
        if k in report and report[k] is not None:
            print(f'  {k:<9} {report[k]:.4f}')
    print(f'  Samples   {report["n_samples"]}')
    print(f'Report saved: {out_dir / "sr_metrics.json"}')


if __name__ == '__main__':
    main()