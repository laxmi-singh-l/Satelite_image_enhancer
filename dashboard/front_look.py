import streamlit as st
from PIL import Image
import numpy as np


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Satellite IR Enhancement & Analysis",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main-header {
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 1rem;
}

.metric-card {
    background: #1e1e1e;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    border: 1px solid #333;
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #00ff88;
}

.metric-label {
    font-size: 0.8rem;
    color: #aaa;
}

.report-box {
    background: #0e1117;
    border: 1px solid #333;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("⚙️ Controls")

    uploaded_file = st.file_uploader(
        "Upload IR Satellite Image",
        type=["png", "jpg", "jpeg", "tif", "tiff"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Input IR Image",
            use_container_width=True
        )

    process_btn = st.button(
        "🚀 Run Full Pipeline",
        type="primary",
        use_container_width=True,
        disabled=(uploaded_file is None)
    )

    st.divider()

    st.markdown("### Sample Images")
    st.info("No samples available. Upload your own IR image.")


# --------------------------------------------------
# MAIN HEADER
# --------------------------------------------------

st.title("🛰️ Satellite IR Enhancement & Analysis System")

st.markdown(
    "Transform low-quality infrared satellite imagery into "
    "enhanced, high-resolution, colorized RGB with semantic analysis."
)


# --------------------------------------------------
# BEFORE IMAGE UPLOAD
# --------------------------------------------------

if uploaded_file is None:

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(
            "📤 **Step 1:** Upload an IR satellite image "
            "using the sidebar."
        )

    with col2:
        st.info(
            "⚙️ **Step 2:** Click **Run Full Pipeline** "
            "to process."
        )

    with col3:
        st.info(
            "📊 **Step 3:** Explore results in the tabs below."
        )

    st.stop()


# --------------------------------------------------
# AFTER IMAGE UPLOAD
# --------------------------------------------------

if not process_btn:

    st.info(
        "Upload an image and click **Run Full Pipeline** to begin."
    )

    st.stop()


# --------------------------------------------------
# DEMO RESULT DATA
# --------------------------------------------------
# No ML models are used.
# These values are only for displaying the frontend.

st.success("Analysis complete! Explore the results below.")


# --------------------------------------------------
# TABS
# --------------------------------------------------

tab_overview, tab_enhance, tab_color, tab_analysis, tab_report = st.tabs([
    "📊 Overview",
    "🔍 Enhancement",
    "🎨 Colorization",
    "🗺️ Analysis",
    "📄 Report"
])


# ==================================================
# OVERVIEW TAB
# ==================================================

with tab_overview:

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Dominant Class",
            "Vegetation"
        )

    with col2:
        st.metric(
            "Natural Coverage",
            "68.5%"
        )

    with col3:
        st.metric(
            "Anthropogenic",
            "31.5%"
        )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Objects Detected",
            "12"
        )

    with col2:
        st.metric(
            "Confidence",
            "91%"
        )

    st.subheader("Scene Description")

    st.markdown("""
    <div class="report-box">
    The satellite scene contains a mixture of vegetation,
    agricultural regions, urban areas and water bodies.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Land Cover Composition")

    chart_data = {
        "Forest": 35,
        "Agriculture": 25,
        "Urban": 18,
        "Water": 12,
        "Barren": 10
    }

    st.bar_chart(chart_data)


# ==================================================
# ENHANCEMENT TAB
# ==================================================

with tab_enhance:

    st.subheader(
        "Infrared Enhancement & Super-Resolution"
    )

    st.markdown(
        "**Input:** 512 × 512 | "
        "**Output:** 2048 × 2048"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            image,
            caption="Original IR",
            use_container_width=True
        )

    with col2:

        # Frontend demo only
        enhanced_image = image.resize(
            (image.width * 2, image.height * 2)
        )

        st.image(
            enhanced_image,
            caption="Enhanced IR",
            use_container_width=True
        )


# ==================================================
# COLORIZATION TAB
# ==================================================

with tab_color:

    st.subheader("IR → RGB Colorization")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.image(
            image,
            caption="Enhanced IR (Input)",
            use_container_width=True
        )

    with col2:

        # Demo RGB representation
        rgb_image = image.convert("RGB")

        st.image(
            rgb_image,
            caption="Colorized RGB (Output)",
            use_container_width=True
        )

    with col3:

        st.image(
            rgb_image,
            caption="Segmentation Overlay",
            use_container_width=True
        )


# ==================================================
# ANALYSIS TAB
# ==================================================

with tab_analysis:

    st.subheader("Land Cover Segmentation")

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            rgb_image,
            caption="Segmentation Map",
            use_container_width=True
        )

    with col2:

        st.image(
            rgb_image,
            caption="Overlay on RGB",
            use_container_width=True
        )

    st.subheader("Legend")

    legend_cols = st.columns(4)

    classes = [
        "🌊 Water",
        "🌲 Forest",
        "🌾 Agriculture",
        "🏙️ Urban",
        "🏜️ Barren",
        "💧 Wetland",
        "🌱 Grassland"
    ]

    for i, name in enumerate(classes):

        with legend_cols[i % 4]:

            st.markdown(name)

    st.subheader("Detected Objects")

    objects = [
        {"Class": "Vehicle", "Confidence": "0.94"},
        {"Class": "Building", "Confidence": "0.91"},
        {"Class": "Road", "Confidence": "0.89"},
        {"Class": "Vehicle", "Confidence": "0.87"}
    ]

    st.dataframe(
        objects,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Land Cover Statistics")

    land_cover = [
        {
            "Class": "Forest",
            "Coverage (%)": 35,
            "Pixels": 92000
        },
        {
            "Class": "Agriculture",
            "Coverage (%)": 25,
            "Pixels": 65000
        },
        {
            "Class": "Urban",
            "Coverage (%)": 18,
            "Pixels": 47000
        },
        {
            "Class": "Water",
            "Coverage (%)": 12,
            "Pixels": 31000
        },
        {
            "Class": "Barren",
            "Coverage (%)": 10,
            "Pixels": 26000
        }
    ]

    st.dataframe(
        land_cover,
        use_container_width=True,
        hide_index=True
    )


# ==================================================
# REPORT TAB
# ==================================================

with tab_report:

    st.subheader("Scene Analysis Report")

    report = {
        "scene_type": "Mixed Terrain",
        "confidence": 0.91,
        "analysis_time": "Demo",
        "classes_detected": 5,
        "objects_detected": 12
    }

    st.json(report)

    st.subheader("Natural Language Description")

    st.markdown("""
    <div class="report-box"
         style="font-size:1.1rem;line-height:1.6;">

    The scene contains forest, agricultural land,
    urban regions and water bodies. Several objects
    including vehicles and buildings were detected.

    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("**Scene Metadata**")

        st.markdown("""
        - **Scene Type:** Mixed Terrain
        - **Confidence:** 91%
        - **Analysis Time:** Demo
        - **Classes Detected:** 5
        """)

    with col2:

        st.markdown("**Object Detection Summary**")

        st.markdown("""
        - **Vehicle:** 7
        - **Building:** 3
        - **Road:** 2
        """)

    st.download_button(
        label="📥 Download Report (JSON)",
        data=str(report),
        file_name="scene_report.json",
        mime="application/json",
        use_container_width=True
    )

    st.download_button(
        label="📥 Download Colorized RGB (PNG)",
        data=open(
            "sample_output.png",
            "rb"
        ).read() if False else b"",
        file_name="colorized_rgb.png",
        mime="image/png",
        use_container_width=True
    )
