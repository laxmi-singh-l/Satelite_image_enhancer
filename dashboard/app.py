import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import numpy as np
import cv2
import json
import time
from io import BytesIO
from PIL import Image

from models import IRSuperResolution, IR2RGB, LandCoverSegmenter
from analysis import SceneAnalyzer
from utils import (
    load_image, save_image,
    create_comparison_view, create_segmentation_overlay,
    colorize_segmentation_mask, draw_detections,
)

st.set_page_config(
    page_title='Satellite IR Enhancement & Analysis',
    page_icon='🛰️',
    layout='wide',
    initial_sidebar_state='expanded',
)

st.markdown("""
<style>
    .main-header { font-size: 2rem; font-weight: 700; margin-bottom: 1rem; }
    .metric-card {
        background: #1e1e1e; border-radius: 10px; padding: 1rem;
        text-align: center; border: 1px solid #333;
    }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #00ff88; }
    .metric-label { font-size: 0.8rem; color: #aaa; }
    .report-box {
        background: #0e1117; border: 1px solid #333;
        border-radius: 8px; padding: 1rem; margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_enhancer():
    return IRSuperResolution(scale=4)


@st.cache_resource
def load_colorizer():
    return IR2RGB()


@st.cache_resource
def load_segmenter():
    return LandCoverSegmenter()


@st.cache_resource
def load_analyzer():
    return SceneAnalyzer()


@st.cache_resource
def load_sr():
    from models import SatelliteSuperResolution
    for ck in ('checkpoints/satelite_sr_best.pt', 'checkpoints/satelite_sr.pt'):
        if Path(ck).exists():
            try:
                return SatelliteSuperResolution(checkpoint=ck)
            except Exception:
                continue
    return None


def process_pipeline(image: np.ndarray) -> dict:
    enhancer = load_enhancer()
    colorizer = load_colorizer()
    segmenter = load_segmenter()
    analyzer = load_analyzer()

    progress_bar = st.progress(0, text='Enhancing IR image...')
    enhanced = enhancer(image)
    progress_bar.progress(25, text='Enhancement complete. Analyzing land cover...')

    analysis = segmenter.analyze(enhanced)
    progress_bar.progress(50, text='Land cover analysis complete. Colorizing...')

    seg_mask = analysis['segmentation_mask']
    rgb = colorizer(enhanced, seg_mask)
    progress_bar.progress(75, text='Colorization complete. Generating report...')

    report = analyzer(
        land_cover=analysis['land_cover'],
        objects=analysis['objects'],
        rgb_image=rgb,
    )
    progress_bar.progress(100, text='Analysis complete!')
    time.sleep(0.3)
    progress_bar.empty()

    return {
        'enhanced': enhanced,
        'rgb': rgb,
        'segmentation_mask': seg_mask,
        'land_cover': analysis['land_cover'],
        'objects': analysis['objects'],
        'report': report,
    }


def main():
    st.title('🛰️ Satellite IR Enhancement & Analysis System')
    st.markdown(
        'Transform low-quality infrared satellite imagery into enhanced, '
        'high-resolution, colorized RGB with semantic analysis.'
    )

    with st.sidebar:
        st.header('Controls')
        uploaded_file = st.file_uploader(
            'Upload IR Satellite Image',
            type=['png', 'jpg', 'jpeg', 'tif', 'tiff'],
        )

        if uploaded_file is not None:
            file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
            input_image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
            if input_image is None:
                input_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if input_image is not None and input_image.ndim == 3:
                    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
                elif input_image is None:
                    st.error('Failed to decode image. Please upload a valid image file.')
                    return

            st.image(input_image, caption='Input IR Image', use_container_width=True,
                     channels='GRAY' if input_image.ndim == 2 else 'RGB')

        process_btn = st.button(
            '🚀 Run Full Pipeline',
            type='primary',
            use_container_width=True,
            disabled=(uploaded_file is None),
        )

        st.divider()
        st.markdown('### Sample Images')
        st.markdown('No samples available. Upload your own IR image.')

    if uploaded_file is None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info('📤 **Step 1:** Upload an IR satellite image using the sidebar.')
        with col2:
            st.info('⚙️ **Step 2:** Click "Run Full Pipeline" to process.')
        with col3:
            st.info('📊 **Step 3:** Explore results in the tabs below.')
        return

    if not process_btn:
        st.info('Upload an image and click **Run Full Pipeline** to begin.')
        return

    with st.spinner('Processing...'):
        results = process_pipeline(input_image)

    st.success('Analysis complete! Explore the results below.')

    tab_overview, tab_enhance, tab_color, tab_analysis, tab_report, tab_sr = st.tabs([
        '📊 Overview', '🔍 Enhancement', '🎨 Colorization',
        '🗺️ Analysis', '📄 Report', '🛰️ SR & Uncertainty',
    ])

    with tab_overview:
        col1, col2, col3 = st.columns(3)
        with col1:
            dom = results['report'].summary_stats
            st.metric('Dominant Class', dom.get('dominant_class', 'N/A').title())
        with col2:
            st.metric('Natural Coverage', f"{dom.get('natural_coverage', 0):.1f}%")
        with col3:
            st.metric('Anthropogenic', f"{dom.get('anthropogenic_coverage', 0):.1f}%")

        col1, col2 = st.columns(2)
        with col1:
            st.metric('Objects Detected', len(results['objects']))
        with col2:
            st.metric('Confidence', f"{results['report'].confidence * 100:.0f}%")

        st.subheader('Scene Description')
        st.markdown(
            f'<div class="report-box">{results["report"].description}</div>',
            unsafe_allow_html=True,
        )

        st.subheader('Land Cover Composition')
        lc = results['land_cover']
        if lc:
            chart_data = {
                k: v['percentage']
                for k, v in sorted(lc.items(), key=lambda x: -x[1]['percentage'])
            }
            st.bar_chart(chart_data)

    with tab_enhance:
        st.subheader('Infrared Enhancement & Super-Resolution')
        st.markdown(
            f'**Input:** {input_image.shape[1]}×{input_image.shape[0]} | '
            f'**Output:** {results["enhanced"].shape[1]}×{results["enhanced"].shape[0]}'
        )

        comparison = create_comparison_view(input_image, results['enhanced'])
        st.image(comparison, caption='Before (left) vs After (right) enhancement',
                 use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.image(input_image, caption='Original IR', use_container_width=True,
                     channels='GRAY')
        with col2:
            st.image(results['enhanced'], caption='Enhanced IR', use_container_width=True,
                     channels='GRAY')

    with tab_color:
        st.subheader('IR → RGB Colorization')

        seg_overlay = create_segmentation_overlay(
            results['enhanced'], results['segmentation_mask'], alpha=0.5
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(results['enhanced'], caption='Enhanced IR (Input)',
                     use_container_width=True, channels='GRAY')
        with col2:
            st.image(results['rgb'], caption='Colorized RGB (Output)',
                     use_container_width=True)
        with col3:
            st.image(seg_overlay, caption='Segmentation Overlay',
                     use_container_width=True)

    with tab_analysis:
        st.subheader('Land Cover Segmentation')

        seg_colored = colorize_segmentation_mask(results['segmentation_mask'])
        overlay = create_segmentation_overlay(
            results['rgb'], results['segmentation_mask'], alpha=0.4
        )

        col1, col2 = st.columns(2)
        with col1:
            st.image(seg_colored, caption='Segmentation Map',
                     use_container_width=True)
        with col2:
            st.image(overlay, caption='Overlay on RGB',
                     use_container_width=True)

        st.subheader('Legend')
        legend_cols = st.columns(4)
        legend_data = [
            (0, ('water', (60, 119, 181))),
            (1, ('forest', (34, 139, 34))),
            (2, ('agriculture', (154, 205, 50))),
            (3, ('urban', (178, 34, 34))),
            (4, ('barren', (210, 180, 140))),
            (5, ('wetland', (0, 206, 209))),
            (6, ('grassland', (124, 252, 0))),
        ]
        for i, (class_id, (name, color)) in enumerate(legend_data):
            with legend_cols[i % 4]:
                r, g, b = color
                st.markdown(
                    f'<span style="display:inline-block;width:12px;height:12px;'
                    f'background:rgb({r},{g},{b});border-radius:2px;margin-right:4px;">'
                    f'</span> {name.title()}',
                    unsafe_allow_html=True,
                )

        if results['objects']:
            st.subheader('Detected Objects')
            obj_draw = draw_detections(results['rgb'], results['objects'])
            st.image(obj_draw, caption=f'{len(results["objects"])} objects detected',
                     use_container_width=True)

            obj_data = []
            for obj in results['objects']:
                obj_data.append({
                    'Class': obj['class'],
                    'Confidence': f"{obj['confidence']:.2f}",
                })
            st.dataframe(obj_data, use_container_width=True, hide_index=True)
        else:
            st.info('No objects detected in this scene (fallback mode).')

        st.subheader('Land Cover Statistics')
        lc_data = []
        for name, stats in results['land_cover'].items():
            lc_data.append({
                'Class': name.title(),
                'Coverage (%)': stats['percentage'],
                'Pixels': stats['pixel_count'],
            })
        st.dataframe(lc_data, use_container_width=True, hide_index=True)

    with tab_report:
        report = results['report']
        st.subheader('Scene Analysis Report')

        st.json(report.to_dict())

        st.subheader('Natural Language Description')
        st.markdown(
            f'<div class="report-box" style="font-size:1.1rem;line-height:1.6;">'
            f'{report.description}</div>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('**Scene Metadata**')
            st.markdown(f'- **Scene Type:** {report.scene_type.title()}')
            st.markdown(f'- **Confidence:** {report.confidence * 100:.0f}%')
            st.markdown(f'- **Analysis Time:** {report.time_analysis}')
            st.markdown(f'- **Classes Detected:** {len(report.land_cover)}')

        with col2:
            st.markdown('**Object Detection Summary**')
            if report.object_counts:
                for cls, count in report.object_counts.items():
                    st.markdown(f'- **{cls}:** {count}')
            else:
                st.markdown('No objects detected.')

        report_json = report.to_json()
        st.download_button(
            label='📥 Download Report (JSON)',
            data=report_json,
            file_name='scene_report.json',
            mime='application/json',
            use_container_width=True,
        )

        buf = BytesIO()
        img_pil = Image.fromarray(results['rgb'])
        img_pil.save(buf, format='PNG')
        st.download_button(
            label='📥 Download Colorized RGB (PNG)',
            data=buf.getvalue(),
            file_name='colorized_rgb.png',
            mime='image/png',
            use_container_width=True,
        )

    with tab_sr:
        st.subheader('Paired Super-Resolution (10 m → <4 m)')
        st.markdown(
            'Uses a trained generative/CNN model to sharpen medium-resolution '
            'satellite imagery. Checkpoints are produced by '
            '`python -m training.train`.'
        )
        enhancer = load_sr()
        if enhancer is None:
            st.warning(
                'No trained SR checkpoint found in `checkpoints/` '
                '(expected `satelite_sr_best.pt` / `satelite_sr.pt`). '
                'Train one with `python -m training.train`, or use the '
                '**Enhancement** tab (OpenCV fallback).'
            )
            st.info(
                'Recommended: `python -m training.train --images-dir data/ --scale 4 '
                '--model esrgan --dropout 0.05 --spectral 0.1 --epochs 60`'
            )
        else:
            sr_input = input_image
            if sr_input.ndim == 2:
                sr_input = cv2.cvtColor(sr_input, cv2.COLOR_GRAY2RGB)
            scale = int(enhancer.scale)

            with st.spinner('Super-resolving...'):
                if enhancer.dropout > 0:
                    enhanced, std_map = enhancer.enhance_with_uncertainty(
                        sr_input, n_samples=10)
                else:
                    enhanced = enhancer(sr_input)
                    std_map = None

            st.markdown(f'**Model:** `{enhancer.model_type}` | '
                        f'**Scale:** {scale}x | '
                        f'**Dropout (uncertainty):** '
                        f'{"enabled" if enhancer.dropout > 0 else "disabled"}')

            st.subheader('Super-Resolution Output')
            comparison = create_comparison_view(sr_input, enhanced)
            st.image(comparison, caption='Input (left) vs Super-resolved (right)',
                     use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.image(sr_input, caption=f'Input LR ({sr_input.shape[1]}×{sr_input.shape[0]})',
                         use_container_width=True)
            with col2:
                st.image(enhanced, caption=f'SR Output ({enhanced.shape[1]}×{enhanced.shape[0]})',
                         use_container_width=True)

            if std_map is not None:
                st.subheader('Uncertainty (MC-Dropout)')
                from models.uncertainty import summarize_uncertainty, aggregate_uncertainty
                un_summary = summarize_uncertainty(std_map, threshold=0.05)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric('Mean Std', f'{un_summary["mean_std"]:.4f}')
                c2.metric('95th Pct Std', f'{un_summary["p95_std"]:.4f}')
                c3.metric('Low-Confidence Pixels', f'{un_summary["low_confidence_pct"]:.1f}%')
                c4.metric('Max Std', f'{un_summary["max_std"]:.4f}')

                std_agg = aggregate_uncertainty(std_map)
                heat = cv2.applyColorMap(
                    (np.clip(std_agg * 255, 0, 255)).astype('uint8'),
                    cv2.COLORMAP_JET,
                )
                st.image(heat, caption='Uncertainty heatmap (jet = high uncertainty)',
                         use_container_width=True, channels='RGB')
                st.caption(
                    'High variance regions mark detail that is inferred by the '
                    'model rather than observed — treat those areas cautiously '
                    'for change detection / damage assessment.'
                )
            else:
                st.info(
                    'Model was trained without `--dropout`, so uncertainty maps '
                    'are unavailable. Retrain with `--dropout 0.05` to enable '
                    'MC-dropout uncertainty.'
                )


if __name__ == '__main__':
    main()
