"""
Streamlit sidebar components.
Provides model selection, parameter controls, and webcam controls.
"""
import streamlit as st
from config.settings import (
    AVAILABLE_MODELS,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IOU_THRESHOLD
)


def render_sidebar():
    """
    Render sidebar controls and return settings.

    Returns:
        Dictionary with user settings
    """
    with st.sidebar:
        st.markdown("""
        <h1 style='text-align: center; color: #000000; font-size: 2rem; margin-bottom: 0.5rem;'>
            ⚙️ Settings
        </h1>
        <p style='text-align: center; color: #666666; font-size: 0.85rem; margin-bottom: 2rem;'>
            Configure your detection system
        </p>
        """, unsafe_allow_html=True)

        # Model Selection
        st.markdown("### 🤖 Model Selection")
        selected_model = st.selectbox(
            "YOLO Model",
            options=list(AVAILABLE_MODELS.keys()),
            help="Choose a YOLO model. Smaller models are faster but less accurate."
        )

        # Display model info
        model_info = AVAILABLE_MODELS[selected_model]
        st.markdown(f"""
        <div class="info-tooltip">
        <strong>Size:</strong> {model_info['size']}<br>
        <strong>Speed:</strong> {model_info['speed']}<br>
        <strong>mAP:</strong> {model_info['map']}<br>
        <em>{model_info['description']}</em>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # Camera Source Selection
        st.markdown("### 📹 Tapo Camera")

        # Camera presets
        camera_presets = {
            "Stream 2 (Fast)": "rtsp://georgedelimpasis:v!nmWiQs3Bs@192.168.1.163:554/stream2",
            "Stream 1 (High Quality)": "rtsp://georgedelimpasis:v!nmWiQs3Bs@192.168.1.163:554/stream1",
        }

        # Preset selection
        selected_preset = st.selectbox(
            "Stream Quality",
            options=list(camera_presets.keys()),
            index=0,  # Default to "Stream 2 (Fast)"
            help="Stream 2 is faster for real-time detection, Stream 1 is higher quality"
        )

        camera_source = camera_presets[selected_preset]
        camera_type = "IP Camera (RTSP)"
        resolution = "640p"  # Resolution from RTSP stream

        st.success(f"✅ {selected_preset}")

        st.divider()

        # Detection Parameters
        st.markdown("### 🎯 Detection Parameters")

        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=DEFAULT_CONF_THRESHOLD,
            step=0.05,
            help="Minimum confidence score for detections"
        )

        iou_threshold = st.slider(
            "IOU Threshold",
            min_value=0.1,
            max_value=1.0,
            value=DEFAULT_IOU_THRESHOLD,
            step=0.05,
            help="IoU threshold for Non-Maximum Suppression"
        )

        image_size = st.select_slider(
            "Image Size",
            options=[320, 640, 1280],
            value=640,
            help="Input size for model (larger = more accurate but slower)"
        )

        st.divider()

        # Zone Setup Controls
        st.markdown("### 🏀 Basketball Scoring")

        # Initialize zone setup state
        if 'setup_zones' not in st.session_state:
            st.session_state.setup_zones = False

        if st.button("⚙️ Setup Detection Zones", use_container_width=True):
            st.session_state.setup_zones = True

        # Show zone status
        if 'zone_manager' in st.session_state:
            zone_count = len(st.session_state.zone_manager.zones)
            if zone_count > 0:
                st.success(f"✅ {zone_count} zone(s) configured")
                # Show score if zones are set up
                if 'Zone 1' in st.session_state.zone_manager.zones and \
                   'Zone 2' in st.session_state.zone_manager.zones:
                    score = st.session_state.zone_manager.total_score
                    st.metric("🎯 Total Score", score)

                    # Show a tiny breakdown in the sidebar as well
                    st.markdown(f"<span style='font-size: 0.8rem; color: #666;'>2PT: {st.session_state.zone_manager.score_2pt} | 3PT: {st.session_state.zone_manager.score_3pt}</span>", unsafe_allow_html=True)

                    if st.button("🔄 Reset Score", use_container_width=True):
                        st.session_state.zone_manager.reset_score()
                        st.rerun()
            else:
                st.info("ℹ️ No zones configured")

        st.divider()

        # Webcam Controls
        st.markdown("### 🎮 Controls")

        # Initialize webcam state
        if 'webcam_active' not in st.session_state:
            st.session_state.webcam_active = False

        # Start/Stop buttons
        col1, col2 = st.columns(2)

        with col1:
            if st.button("▶️ Start", use_container_width=True, disabled=st.session_state.webcam_active):
                st.session_state.webcam_active = True
                st.rerun()

        with col2:
            if st.button("⏹️ Stop", use_container_width=True, disabled=not st.session_state.webcam_active):
                st.session_state.webcam_active = False
                st.rerun()

        st.divider()

        # Information
        st.markdown("### 💡 Pro Tips")
        st.markdown("""
        <div style='background: #f5f5f5; border-left: 4px solid #000000; padding: 1rem; border-radius: 8px; font-size: 0.85rem; line-height: 1.8;'>
        <strong style='color: #000000;'>Optimize your detection:</strong><br>
        🎯 Use <strong>Custom Ball</strong> model<br>
        ⚡ Use <strong>Stream 2</strong> for speed<br>
        🔍 Lower confidence to <strong>0.20-0.25</strong><br>
        🏀 Setup zones for <strong>auto-scoring</strong><br>
        💡 Ensure <strong>good lighting</strong>
        </div>
        """, unsafe_allow_html=True)

    # Return settings
    return {
        'model_name': selected_model,
        'conf_threshold': conf_threshold,
        'iou_threshold': iou_threshold,
        'image_size': image_size,
        'resolution': resolution,
        'camera_source': camera_source,
        'camera_type': camera_type,
        'webcam_active': st.session_state.webcam_active
    }
