#!/usr/bin/env python3
"""
Satellite IR Enhancement & Analysis System — CLI Entry Point

Usage:
    python drawing/run.py --image samples/ir_input.png
    python drawing/run.py --image samples/ir_input.png --output results/
    python drawing/run.py --image samples/ir_input.png --report-only
"""

import sys
import argparse
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import cv2
from models import IRSuperResolution, IR2RGB, LandCoverSegmenter
from analysis import SceneAnalyzer
from utils import load_image, save_image, create_segmentation_overlay, colorize_segmentation_mask, draw_detections


def parse_args():
    parser = argparse.ArgumentParser(description='Satellite IR Enhancement & Analysis')
    parser.add_argument('--image', '-i', required=True, help='Path to input IR image')
    parser.add_argument('--output', '-o', default='results', help='Output directory')
    parser.add_argument('--scale', type=int, default=4, help='Super-resolution scale factor')
    parser.add_argument('--report-only', action='store_true', help='Only generate JSON report (no images)')
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.image)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'[1/5] Loading IR image: {input_path}')
    image = load_image(str(input_path))
    print(f'      Shape: {image.shape}')

    print('[2/5] Enhancing IR image (denoise + super-resolution)...')
    enhancer = IRSuperResolution(scale=args.scale)
    enhanced = enhancer(image)
    print(f'      Output: {enhanced.shape}')

    if not args.report_only:
        save_image(enhanced, str(output_dir / '01_enhanced_ir.png'))

    print('[3/5] Analyzing land cover and detecting objects...')
    segmenter = LandCoverSegmenter()
    analysis = segmenter.analyze(enhanced)

    land_cover = analysis['land_cover']
    objects = analysis['objects']
    seg_mask = analysis['segmentation_mask']

    dominant = max(land_cover.items(), key=lambda x: x[1]['percentage'])
    print(f'      Dominant class: {dominant[0]} ({dominant[1]["percentage"]:.1f}%)')
    print(f'      Objects detected: {len(objects)}')

    if not args.report_only:
        seg_colored = colorize_segmentation_mask(seg_mask)
        overlay = create_segmentation_overlay(enhanced, seg_mask)
        save_image(seg_colored, str(output_dir / '02_segmentation_map.png'))
        save_image(overlay, str(output_dir / '03_segmentation_overlay.png'))

        if objects:
            enhanced_rgb = enhanced if enhanced.ndim == 3 else cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
            detections_vis = draw_detections(enhanced_rgb, objects)
            detections_vis = cv2.cvtColor(detections_vis, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_dir / '04_detections.png'), detections_vis)

    print('[4/5] Colorizing IR to RGB...')
    if enhanced.ndim == 2:
        enhanced_vis = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    else:
        enhanced_vis = enhanced.copy()

    colorizer = IR2RGB()
    rgb = colorizer(enhanced_vis, seg_mask)
    print(f'      RGB output: {rgb.shape}')

    if not args.report_only:
        save_image(rgb, str(output_dir / '05_colorized_rgb.png'))

    print('[5/5] Generating scene analysis report...')
    analyzer = SceneAnalyzer()
    report = analyzer(
        land_cover=land_cover,
        objects=objects,
        rgb_image=rgb,
    )

    report_path = output_dir / 'report.json'
    with open(report_path, 'w') as f:
        f.write(report.to_json())
    print(f'      Report saved: {report_path}')

    print()
    print('=' * 60)
    print('SCENE ANALYSIS REPORT')
    print('=' * 60)
    print(f'Scene Type:     {report.scene_type.title()}')
    print(f'Confidence:     {report.confidence * 100:.0f}%')
    print(f'Time:           {report.time_analysis}')
    print(f'Description:    {report.description[:200]}...')
    print()
    print('Land Cover Composition:')
    for name, stats in sorted(land_cover.items(), key=lambda x: -x[1]['percentage']):
        print(f'  {name:15s} {stats["percentage"]:6.2f}%')
    if objects:
        print(f'\nObjects Detected: {len(objects)}')
        counts = {}
        for obj in objects:
            counts[obj['class']] = counts.get(obj['class'], 0) + 1
        for cls, cnt in counts.items():
            print(f'  {cls}: {cnt}')
    print('=' * 60)
    print(f'Output files saved to: {output_dir.resolve()}')


if __name__ == '__main__':
    main()
