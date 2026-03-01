"""
Streamlit sidebar components.
Provides model selection, parameter controls, and webcam controls.
"""
import streamlit as st
from config.settings import (
    AVAILABLE_MODELS,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    WEBCAM_RESOLUTIONS
)


def render_sidebar():
    """
    Render sidebar controls and return settings.

    Returns:
        Dictionary with user settings
    """
    with st.sidebar:
        st.title("⚙️ Settings")

        # Model Selection
        st.subheader("🤖 Model Selection")
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
        st.subheader("📹 Camera Source")

        camera_type = st.radio(
            "Camera Type",
            options=["Local Webcam", "IP Camera (RTSP)"],
            help="Choose between local webcam or IP camera via RTSP"
        )

        if camera_type == "Local Webcam":
            camera_id = st.number_input(
                "Camera ID",
                min_value=0,
                max_value=10,
                value=0,
                help="0 for default camera, 1+ for additional cameras"
            )
            camera_source = int(camera_id)
        else:
            # IP Camera RTSP configuration
            st.markdown("**RTSP Configuration**")

            # Camera presets
            camera_presets = {
                "Tapo Camera": "rtsp://georgedelimpasis:v!nmWiQs3Bs@192.168.1.163:554/stream2",
                "Tapo Camera (High Quality)": "rtsp://georgedelimpasis:v!nmWiQs3Bs@192.168.1.163:554/stream1",
                "Custom URL": ""
            }

            # Preset selection
            selected_preset = st.selectbox(
                "Camera Preset",
                options=list(camera_presets.keys()),
                index=0,  # Default to "Tapo Camera"
                help="Select a saved camera or choose Custom URL to enter your own"
            )

            # Initialize RTSP URL in session state
            if 'rtsp_url' not in st.session_state:
                st.session_state.rtsp_url = camera_presets["Tapo Camera"]

            # If a preset is selected, use it
            if selected_preset != "Custom URL":
                rtsp_url = camera_presets[selected_preset]
                st.session_state.rtsp_url = rtsp_url
                st.success(f"✅ Using: {selected_preset}")
            else:
                # Custom URL input
                rtsp_url = st.text_input(
                    "RTSP URL",
                    value=st.session_state.rtsp_url,
                    help="Format: rtsp://username:password@ip_address:554/stream1",
                    placeholder="rtsp://admin:password@192.168.1.100:554/stream1"
                )

            # Save to session state
            st.session_state.rtsp_url = rtsp_url
            camera_source = rtsp_url

            # RTSP URL examples
            with st.expander("📖 RTSP URL Examples"):
                st.markdown("""
                **Tapo C220:**
                - Main stream: `rtsp://user:pass@ip:554/stream1`
                - Sub stream: `rtsp://user:pass@ip:554/stream2`

                **Tips:**
                - Find IP in Tapo app: Device Settings > Device Info
                - Default port: 554
                - Use main camera account credentials
                - Test stream1 first (better quality)
                - stream2 is lower quality but faster
                """)

        st.divider()

        # Detection Parameters
        st.subheader("🎯 Detection Parameters")

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

        # Resolution Settings (only for local webcam)
        st.subheader("🎬 Stream Settings")

        if camera_type == "Local Webcam":
            resolution = st.selectbox(
                "Resolution",
                options=list(WEBCAM_RESOLUTIONS.keys()),
                index=1,  # Default to 640p
                help="Webcam resolution (lower = faster processing)"
            )
        else:
            resolution = "640p"  # Default for RTSP, actual resolution from stream
            st.info("ℹ️ Resolution is determined by the RTSP stream")

        st.divider()

        # Webcam Controls
        st.subheader("🎮 Controls")

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
        st.subheader("ℹ️ Info")
        st.markdown("""
        **Tips for best performance:**
        - Use YOLOv8n for real-time detection
        - Lower resolution for faster FPS
        - Adjust confidence threshold to reduce false positives
        - Ensure good lighting for better accuracy
        """)

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
