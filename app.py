"""
YOLO Real-Time Webcam Detection App
Main Streamlit application for object detection using YOLOv8.
"""
import streamlit as st
import time
from src.detector import YOLODetector
from src.webcam import WebcamCapture
from src.utils import (
    generate_class_colors,
    draw_bounding_boxes,
    calculate_fps,
    format_detection_stats,
    add_fps_overlay
)
from ui.sidebar import render_sidebar
from ui.styles import inject_custom_css


# Page configuration
st.set_page_config(
    page_title="YOLO Webcam Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)


def initialize_session_state():
    """Initialize session state variables."""
    if 'detector' not in st.session_state:
        st.session_state.detector = None

    if 'webcam' not in st.session_state:
        st.session_state.webcam = None

    if 'webcam_active' not in st.session_state:
        st.session_state.webcam_active = False

    if 'current_stats' not in st.session_state:
        st.session_state.current_stats = None

    if 'current_fps' not in st.session_state:
        st.session_state.current_fps = 0.0

    if 'class_colors' not in st.session_state:
        st.session_state.class_colors = generate_class_colors()

    if 'current_model' not in st.session_state:
        st.session_state.current_model = None


def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()

    # Inject custom CSS
    inject_custom_css()

    # Render sidebar and get settings
    settings = render_sidebar()

    # Main content area
    st.markdown('<h1 class="main-title">🎯 YOLO Real-Time Webcam Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Detect objects in real-time using YOLOv8</p>', unsafe_allow_html=True)

    # Initialize or update detector if model changed
    if st.session_state.current_model != settings['model_name']:
        with st.spinner(f"Loading {settings['model_name']}..."):
            st.session_state.detector = YOLODetector(settings['model_name'])
            st.session_state.current_model = settings['model_name']
            st.success(f"✅ {settings['model_name']} loaded successfully!")

    # Display model info
    if st.session_state.detector:
        model_info = st.session_state.detector.get_model_info()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Model:** {model_info['model_name']}")
        with col2:
            st.info(f"**Device:** {model_info['device'].upper()}")
        with col3:
            st.info(f"**Size:** {model_info['size']}")

    st.divider()

    # Webcam processing
    if settings['webcam_active']:
        # Initialize webcam if needed
        if st.session_state.webcam is None or not st.session_state.webcam.is_active:
            st.session_state.webcam = WebcamCapture(
                camera_source=settings['camera_source'],
                resolution=settings['resolution']
            )
            if not st.session_state.webcam.start():
                error_msg = "❌ Failed to start camera. "
                if settings['camera_type'] == "IP Camera (RTSP)":
                    error_msg += "Please check:\n- RTSP URL is correct\n- Camera is online\n- Username/password are correct\n- Port 554 is accessible"
                else:
                    error_msg += "Please check camera permissions."
                st.error(error_msg)
                st.session_state.webcam_active = False
                st.stop()

        # Create layout: video on left, stats on right
        col1, col2 = st.columns([2, 1])

        with col1:
            video_placeholder = st.empty()

        with col2:
            st.markdown("### 📊 Live Statistics")
            stats_person = st.empty()
            stats_total = st.empty()
            stats_fps = st.empty()
            stats_conf = st.empty()
            st.markdown("---")
            stats_classes = st.empty()

        status_placeholder = st.empty()

        # FPS tracking
        frame_count = 0
        start_time = time.time()

        # Real-time processing loop
        try:
            while st.session_state.webcam_active:
                # Read frame
                frame = st.session_state.webcam.read()

                if frame is None:
                    status_placeholder.warning("⚠️ No frame captured from webcam")
                    break

                # Run detection
                detections = st.session_state.detector.detect(
                    frame,
                    conf_threshold=settings['conf_threshold'],
                    iou_threshold=settings['iou_threshold'],
                    image_size=settings['image_size']
                )

                # Filter to only show persons
                detections = [d for d in detections if d['class_name'] == 'person']

                # Draw bounding boxes
                annotated_frame = draw_bounding_boxes(
                    frame,
                    detections,
                    st.session_state.class_colors
                )

                # Calculate and add FPS
                frame_count += 1
                fps = calculate_fps(start_time, frame_count)
                st.session_state.current_fps = fps
                annotated_frame = add_fps_overlay(annotated_frame, fps)

                # Calculate statistics
                stats = format_detection_stats(detections)
                st.session_state.current_stats = stats

                # Update real-time statistics display
                person_count = stats.get('class_counts', {}).get('person', 0)
                stats_person.metric("👥 Persons Detected", person_count, help="Number of people currently visible")
                stats_total.metric("📦 Total Objects", stats.get('total_objects', 0))
                fps_color = "🟢" if fps >= 10 else "🟡" if fps >= 5 else "🔴"
                stats_fps.metric("⚡ FPS", f"{fps_color} {fps:.1f}")
                avg_conf = stats.get('avg_confidence', 0)
                stats_conf.metric("🎯 Avg Confidence", f"{avg_conf:.1%}" if avg_conf > 0 else "N/A")

                # Update class breakdown
                class_counts = stats.get('class_counts', {})
                if class_counts:
                    class_text = "**Detected Classes:**\n\n"
                    for class_name, count in class_counts.items():
                        if class_name == 'person':
                            class_text += f"👤 **{class_name}: {count}**\n\n"
                        else:
                            class_text += f"• {class_name}: {count}\n\n"
                    stats_classes.markdown(class_text)
                else:
                    stats_classes.info("Waiting for detections...")

                # Display frame
                video_placeholder.image(
                    annotated_frame,
                    channels="BGR",
                    use_column_width=True
                )

                # Check if stop button was pressed
                if not st.session_state.webcam_active:
                    break

                # Small delay to prevent UI overload
                time.sleep(0.01)

        except Exception as e:
            st.error(f"❌ Error during webcam processing: {e}")

        finally:
            # Cleanup
            if st.session_state.webcam is not None:
                st.session_state.webcam.stop()
                st.session_state.webcam = None

    else:
        # Webcam not active - show instructions
        st.markdown("""
        <div class="detection-box">
            <h2>👋 Welcome to YOLO Webcam Detection!</h2>
            <p>To get started:</p>
            <ol>
                <li>Choose a YOLO model from the sidebar (YOLOv8n recommended for real-time)</li>
                <li>Adjust detection parameters if needed</li>
                <li>Select your preferred webcam resolution</li>
                <li>Click the <strong>▶️ Start</strong> button to begin detection</li>
            </ol>
            <br>
            <p><strong>💡 Tip:</strong> For the best real-time performance, use YOLOv8n with 640p resolution.</p>
        </div>
        """, unsafe_allow_html=True)

        # Cleanup webcam if it exists
        if st.session_state.webcam is not None:
            st.session_state.webcam.stop()
            st.session_state.webcam = None

    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        Built with Streamlit, YOLOv8, and OpenCV | Real-time Object Detection
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
