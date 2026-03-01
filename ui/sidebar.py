"""
Streamlit sidebar components.
Provides model selection, parameter controls, and webcam controls.
"""
import streamlit as st
from config.settings import (
    AVAILABLE_MODELS,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    IP_CAMERA_RESOLUTIONS
)
from src.ip_camera_discovery import IPCameraDiscovery, CameraDevice


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
        st.markdown("### 📹 Camera Source")

        # Initialize camera discovery in session state
        if 'camera_discovery' not in st.session_state:
            st.session_state.camera_discovery = IPCameraDiscovery()
            st.session_state.discovered_cameras = []
            st.session_state.scanning = False

        # Static camera presets (for RTSP cameras and fallback)
        static_presets = {
            "IP Webcam (Phone)": "http://192.168.1.136:8080/video",
            "Phone Camera (OBS USB)": 2,
            "Stream 2 (Fast)": "rtsp://georgedelimpasis:v!nmWiQs3Bs@192.168.1.163:554/stream2",
            "Stream 1 (High Quality)": "rtsp://georgedelimpasis:v!nmWiQs3Bs@192.168.1.163:554/stream1",
        }

        # Network scan button
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("**IP Camera Discovery**")
        with col2:
            if st.button("🔍 Scan", use_container_width=True, disabled=st.session_state.scanning):
                st.session_state.scanning = True
                with st.spinner("Scanning network..."):
                    st.session_state.discovered_cameras = st.session_state.camera_discovery.scan_network()
                st.session_state.scanning = False
                st.rerun()

        # Build camera options dictionary
        camera_options = {}

        # Add discovered IP cameras first
        if st.session_state.discovered_cameras:
            for camera in st.session_state.discovered_cameras:
                camera_options[f"📱 {camera.name}"] = ('discovered', camera)

        # Add static presets
        for name, source in static_presets.items():
            camera_options[name] = ('static', source)

        # Add manual entry option
        camera_options["✏️ Manual IP Entry"] = ('manual', None)

        # Camera selection
        if camera_options:
            selected_camera_name = st.selectbox(
                "Select Camera",
                options=list(camera_options.keys()),
                help="Discovered IP Webcam devices and saved presets"
            )

            camera_type, camera_data = camera_options[selected_camera_name]

            # Handle different camera types
            if camera_type == 'discovered':
                # Discovered IP camera
                camera_device: CameraDevice = camera_data

                # Resolution selection
                ip_resolution = st.select_slider(
                    "IP Camera Resolution",
                    options=list(IP_CAMERA_RESOLUTIONS.keys()),
                    value="480p",
                    help="Lower resolution = higher FPS and lower latency"
                )

                # Build camera URL (always use MJPEG)
                resolution_param = IP_CAMERA_RESOLUTIONS[ip_resolution]
                camera_source = camera_device.get_url('mjpeg', resolution_param)
                streaming_mode_value = 'mjpeg'
                resolution = "640p"  # For internal processing

                st.success(f"✅ {camera_device.name} (MJPEG)")

            elif camera_type == 'manual':
                # Manual IP entry
                st.markdown("**Manual Configuration**")
                manual_ip = st.text_input("IP Address", "192.168.1.136", help="IP address of your IP Webcam device")
                manual_port = st.number_input("Port", value=8080, min_value=1, max_value=65535)

                # Resolution selection
                ip_resolution = st.select_slider(
                    "IP Camera Resolution",
                    options=list(IP_CAMERA_RESOLUTIONS.keys()),
                    value="480p",
                    help="Lower resolution = higher FPS"
                )

                # Build URL (always MJPEG)
                resolution_param = IP_CAMERA_RESOLUTIONS[ip_resolution]
                camera_source = f"http://{manual_ip}:{manual_port}/video?{resolution_param}"
                streaming_mode_value = 'mjpeg'
                resolution = "640p"

                st.info(f"📡 Using: {camera_source}")

            else:
                # Static preset (RTSP or local camera)
                camera_source = camera_data
                streaming_mode_value = 'mjpeg'  # RTSP uses standard mode
                resolution = "640p"

                st.success(f"✅ {selected_camera_name}")

        else:
            # Fallback
            camera_source = 2
            streaming_mode_value = 'mjpeg'
            resolution = "640p"

        # Start/Stop Controls
        st.markdown("")  # Small spacing
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Start", use_container_width=True, disabled=st.session_state.get('webcam_active', False)):
                st.session_state.webcam_active = True
                st.rerun()
        with col2:
            if st.button("⏹️ Stop", use_container_width=True, disabled=not st.session_state.get('webcam_active', False)):
                st.session_state.webcam_active = False
                st.rerun()

        # Connection status (if active)
        if st.session_state.get('webcam_active') and st.session_state.get('webcam'):
            webcam = st.session_state.webcam
            fps = webcam.get_current_fps()
            latency = webcam.latency_ms

            # Color-coded status based on performance
            if fps >= 20:
                st.success(f"🟢 Connected: {fps:.1f} FPS | {latency:.0f}ms latency")
            elif fps >= 10:
                st.warning(f"🟡 Connected: {fps:.1f} FPS | {latency:.0f}ms latency")
            else:
                st.error(f"🔴 Slow: {fps:.1f} FPS | {latency:.0f}ms latency")

        # Advanced settings expander
        with st.expander("⚙️ Advanced Settings & Tips"):
            st.markdown("**Performance Optimization**")
            st.markdown("""
            - **480p resolution**: Best balance for ball detection
            - **5GHz WiFi**: Faster than 2.4GHz
            - **Same subnet**: Phone and computer on same network
            - **Close apps**: Free up phone resources
            - **Lower quality**: 50-60% in IP Webcam app settings
            """)

            st.markdown("**IP Webcam App Settings**")
            st.markdown("""
            - Video resolution: 854×480 (480p)
            - Quality: 50-60%
            - FPS limit: 30
            - Enable: "Prevent phone from sleeping"
            - Disable: Audio (not needed)
            """)

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
        'streaming_mode': streaming_mode_value,
        'webcam_active': st.session_state.webcam_active
    }
