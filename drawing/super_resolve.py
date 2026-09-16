#!/usr/bin/env python3
"""End-to-end satellite super-resolution inference.

Takes a medium-resolution satellite image (e.g. 10 m Sentinel-2 GeoTIFF,
RGB or multi-band), runs the trained generative/CNN super-resolution model,
optionally estimates Monte-Carlo dropout uncertainty, and writes:

- ``<out>/enhanced_<scale>x.tif``   — georeferenced enhanced product (CRS +
  geotransform preserved, pixel footprint scaled by the SR factor)
- ``<out>/uncertainty.tif``         — per-pixel per-band std map (+ PNG heatmap)
- ``<out>/uncertainty_summary.json``— mean/p95/max std & low-confidence %
- ``<out>/sr_summary.json``         — run metadata, input/output resolution
- ``<out>/report.json`` (+ overlays, ``--analyze``)
  — land-cover segmentation / object detection / scene report on the enhanced
  product (crop monitoring, urban analysis, disaster assessment)

Usage:

    python drawing/super_resolve.py --input scene_sentinel2.tif \\
        --checkpoint checkpoints/satelite_sr_best.pt --uncertainty 20 --analyze
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models import SatelliteSuperResolution
from models.uncertainty import summarize_uncertainty, aggregate_uncertainty
from analysis import SceneAnalyzer
from utils.geo import read_geo_image, write_geo_image, pixel_resolution
from utils import create_segmentation_overlay, colorize_segmentation_mask, draw_detections


def parse_args():
    p = argparse.ArgumentParser(description='Satellite super-resolution inference')
    p.add_argument('--input', '-i', required=True, help='Input GeoTIFF / image')
    p.add_argument('--checkpoint', '-c', type=str,
                   default='checkpoints/satelite_sr_best.pt',
                   help='Trained SR checkpoint')
    p.add_argument('--out', '-o', type=str, default='results/super_resolved')
    p.add_argument('--uncertainty', type=int, default=0, metavar='N',
                   help='Run MC-dropout uncertainty with N samples (>0 to enable)')
    p.add_argument('--threshold', type=float, default=0.05,
                   help='Low-confidence std threshold (normalised domain)')
    p.add_argument('--analyze', action='store_true',
                   help='Run land-cover / object / scene analysis on the enhanced product')
    p.add_argument('--input-res', type=float, default=None,
                   help='Input pixel size in meters (default: from geotransform, else 10.0)')
    p.add_argument('--device', type=str, default=None)
    return p.parse_args()


def _to_rgb_or_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] >= 3:
        return img[..., :3]
    if img.ndim == 3 and img.shape[2] == 1:
        return img[..., 0]
    return img


def run_analysis(enhanced: np.ndarray, out_dir: Path) -> dict:
    from models import LandCoverSegmenter

    img = _to_rgb_or_gray(enhanced)
    if img.ndim == 2:
        img_vis = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img_vis = img

    segmenter = LandCoverSegmenter()
    analysis = segmenter.analyze(img_vis)

    seg_colored = colorize_segmentation_mask(analysis['segmentation_mask'])
    overlay = create_segmentation_overlay(img_vis, analysis['segmentation_mask'], alpha=0.4)
    cv2.imwrite(str(out_dir / 'segmentation_map.png'),
                cv2.cvtColor(seg_colored, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / 'segmentation_overlay.png'),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    detections = analysis['objects']
    if detections:
        det_vis = draw_detections(img_vis, detections)
        cv2.imwrite(str(out_dir / 'detections.png'),
                    cv2.cvtColor(det_vis, cv2.COLOR_RGB2BGR))

    analyzer = SceneAnalyzer()
    report = analyzer(
        land_cover=analysis['land_cover'],
        objects=detections,
        rgb_image=img_vis,
    )
    with open(out_dir / 'report.json', 'w') as f:
        f.write(report.to_json())
    return report.to_dict()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'[1/4] Reading input: {args.input}')
    image, meta = read_geo_image(args.input)
    bands = 1 if image.ndim == 2 else image.shape[2]
    in_res = args.input_res or pixel_resolution(meta, default_m=10.0)
    print(f'      Shape: {image.shape}, bands: {bands}, input res: {in_res:.1f} m')

    print('[2/4] Loading super-resolution model...')
    enhancer = SatelliteSuperResolution(
        n_channels=bands, checkpoint=args.checkpoint, device=args.device,
    )
    if enhancer.model is None:
        raise RuntimeError(f'Could not load checkpoint: {args.checkpoint}')
    scale = enhancer.scale
    out_res = in_res / scale
    print(f'      Model: {enhancer.model_type}, scale: {scale}x, '
          f'output res: {out_res:.2f} m')

    print('[3/4] Super-resolving...')
    enhanced = enhancer(image)
    print(f'      Output: {enhanced.shape}')

    summary = {
        'input': str(args.input),
        'input_shape': list(image.shape),
        'bands': bands,
        'input_res_m': round(in_res, 3),
        'scale': scale,
        'output_res_m': round(out_res, 3),
        'output_meets_sub4m': out_res < 4.0,
        'model': enhancer.model_type,
        'checkpoint': str(args.checkpoint),
        'enhanced': str(out_dir / f'enhanced_{scale}x.tif'),
    }

    if args.uncertainty > 0:
        print(f'      Estimating MC-dropout uncertainty ({args.uncertainty} samples)...')
        mean_u8, std = enhancer.enhance_with_uncertainty(
            image, n_samples=args.uncertainty)
        enhanced = mean_u8
        un_summary = summarize_uncertainty(std, threshold=args.threshold)
        summary['uncertainty'] = un_summary
        summary['uncertainty_samples'] = args.uncertainty
        summary['uncertainty_threshold'] = args.threshold
        print(f'      Uncertainty: mean std {un_summary["mean_std"]:.4f}, '
              f'low-confidence pixels {un_summary["low_confidence_pct"]:.1f}%')

        std_agg = aggregate_uncertainty(std, axis=-1) if std.ndim == 3 else std
        write_geo_image(str(out_dir / 'uncertainty.tif'), std, meta, scale,
                        dtype='float32')
        heat = cv2.applyColorMap(
            (np.clip(std_agg * 255, 0, 255)).astype('uint8'), cv2.COLORMAP_JET)
        cv2.imwrite(str(out_dir / 'uncertainty_heatmap.png'), heat)
        print(f'      Saved: {out_dir / "uncertainty.tif"} + heatmap')
    else:
        summary['uncertainty'] = None
        print('      (uncertainty skipped; pass --uncertainty N to enable)')

    print('[4/4] Writing outputs...')
    write_geo_image(str(out_dir / f'enhanced_{scale}x.tif'), enhanced, meta, scale,
                    dtype='uint8')
    summary['output'] = str(out_dir / f'enhanced_{scale}x.tif')

    if args.analyze:
        print('      Analysing enhanced product (land cover / objects)...')
        try:
            report = run_analysis(enhanced, out_dir)
            summary['analysis_report'] = str(out_dir / 'report.json')
            summary['scene_type'] = report.get('scene_type')
        except Exception as exc:  # analysis is best-effort
            print(f'      Analysis skipped: {exc}')

    with open(out_dir / 'sr_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Summary saved: {out_dir / "sr_summary.json"}')
    if out_res < 4.0:
        print(f'SUCCESS: enhanced product at {out_res:.2f} m satisfies <4 m output.')


if __name__ == '__main__':
    main()