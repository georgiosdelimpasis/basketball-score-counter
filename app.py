"""
YOLO Real-Time Webcam Detection App
Main Streamlit application for object detection using YOLOv8.
"""
import streamlit as st
import time
from src.detector import YOLODetector
from src.webcam import WebcamCapture
from src.zones import ZoneManager
from src.utils import (
    generate_class_colors,
    draw_bounding_boxes,
    calculate_fps,
    format_detection_stats,
    draw_zones
)
from ui.sidebar import render_sidebar
from ui.styles import inject_custom_css
from ui.zone_setup import render_zone_setup


# Page configuration
st.set_page_config(
    page_title="🏀 Basketball Detection AI",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Basketball Detection AI powered by YOLOv8 and Streamlit"
    }
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

    if 'zone_manager' not in st.session_state:
        st.session_state.zone_manager = ZoneManager()

    if 'setup_zones' not in st.session_state:
        st.session_state.setup_zones = False

    if 'unconfirmed_scores' not in st.session_state:
        st.session_state.unconfirmed_scores = 0


def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()

    # Inject custom CSS
    inject_custom_css()

    # Render sidebar and get settings
    settings = render_sidebar()

    # Main content area
    st.markdown('<h1 class="main-title">🏀 Basketball Detection AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Real-time ball tracking & automatic scoring powered by YOLOv8</p>', unsafe_allow_html=True)

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
            st.markdown(f"""
            <div class="model-info-card">
                <strong>🤖 Model</strong><br>
                <span style="color: #666666; font-size: 0.9rem;">{model_info['model_name']}</span>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            device_emoji = "⚡" if model_info['device'] == 'mps' else "🖥️" if model_info['device'] == 'cuda' else "💻"
            st.markdown(f"""
            <div class="model-info-card">
                <strong>{device_emoji} Device</strong><br>
                <span style="color: #666666; font-size: 0.9rem;">{model_info['device'].upper()}</span>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="model-info-card">
                <strong>📦 Size</strong><br>
                <span style="color: #666666; font-size: 0.9rem;">{model_info['size']}</span>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # Manual Score Confirmation UI
    if st.session_state.unconfirmed_scores > 0:
        st.warning(f"🎉 **SCORE DETECTED!** ({st.session_state.unconfirmed_scores} pending) Please confirm point value:", icon="🏀")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("✌️ 2 Points", use_container_width=True, type="primary"):
                st.session_state.zone_manager.add_manual_score(2)
                st.session_state.unconfirmed_scores -= 1
                st.rerun()
        with col2:
            if st.button("🔥 3 Points", use_container_width=True, type="primary"):
                st.session_state.zone_manager.add_manual_score(3)
                st.session_state.unconfirmed_scores -= 1
                st.rerun()
        with col3:
            if st.button("❌ Cancel (Miss)", use_container_width=True):
                st.session_state.unconfirmed_scores -= 1
                st.rerun()
        
        st.divider()

    # Zone setup mode
    if st.session_state.setup_zones:
        render_zone_setup(st.session_state.webcam, st.session_state.zone_manager)
        return  # Don't process video while setting up zones

    # Webcam processing
    if settings['webcam_active']:
        # Initialize webcam if needed
        if st.session_state.webcam is None or not st.session_state.webcam.is_active:
            st.session_state.webcam = WebcamCapture(
                camera_source=settings['camera_source'],
                resolution=settings['resolution']
            )
            if not st.session_state.webcam.start():
                st.error("""
                ❌ Failed to connect to Tapo camera

                **Please check:**
                - Camera is powered on and connected to WiFi
                - Camera IP address is correct (192.168.1.163)
                - Username/password are correct
                - Your device is on the same network as the camera
                - Port 554 is accessible

                **Troubleshooting:**
                - Check camera IP in Tapo app: Device Settings → Device Info
                - Try Stream 1 if Stream 2 fails
                - Restart the camera if needed
                """)
                st.session_state.webcam_active = False
                st.stop()

        # Create layout: video on left, stats on right
        col1, col2 = st.columns([2, 1])

        with col1:
            video_placeholder = st.empty()

        with col2:
            st.markdown("### 📊 Live Statistics")
            
            # Show score breakdown
            st.markdown("#### 🏀 Match Score")
            score_total = st.empty()
            score_breakdown = st.empty()
            
            if st.button("🔄 Reset Score"):
                 st.session_state.zone_manager.reset_score()
                 st.success("Score reset!")
            
            st.markdown("---")
            stats_fps = st.empty()
            stats_conf = st.empty()

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

                # Draw bounding boxes
                annotated_frame = draw_bounding_boxes(
                    frame,
                    detections,
                    st.session_state.class_colors
                )

                # Draw zones if configured
                if st.session_state.zone_manager.zones:
                    annotated_frame = draw_zones(annotated_frame, st.session_state.zone_manager)

                # Check for scoring if zones are set up
                if 'Zone 1' in st.session_state.zone_manager.zones and \
                   'Zone 2' in st.session_state.zone_manager.zones:
                    scored = st.session_state.zone_manager.check_scoring(detections)
                    if scored:
                        st.session_state.unconfirmed_scores += 1
                        st.rerun() # Interrupt video loop to ask user for score confirmation

                # Calculate FPS (for stats display only)
                frame_count += 1
                fps = calculate_fps(start_time, frame_count)
                st.session_state.current_fps = fps

                # Calculate statistics
                stats = format_detection_stats(detections)
                st.session_state.current_stats = stats

                # Update real-time statistics display
                
                # Update Score
                score_total.metric("🎯 TOTAL SCORE", st.session_state.zone_manager.total_score)
                score_breakdown.markdown(f"""
                <div style='background-color: #f0f2f6; padding: 15px; border-radius: 8px; margin-bottom: 15px; border: 1px solid #e0e4eb;'>
                    <div style='display: flex; justify-content: space-between; margin-bottom: 8px; color: #1e1e1e;'>
                        <span style='font-size: 1.1em;'>✌️ 2-Pointers:</span> 
                        <strong style='font-size: 1.1em;'>{st.session_state.zone_manager.score_2pt} &times; 2 = <span style='color: #0068c9;'>{st.session_state.zone_manager.score_2pt * 2}</span></strong>
                    </div>
                    <div style='display: flex; justify-content: space-between; color: #1e1e1e;'>
                        <span style='font-size: 1.1em;'>🔥 3-Pointers:</span> 
                        <strong style='font-size: 1.1em;'>{st.session_state.zone_manager.score_3pt} &times; 3 = <span style='color: #ff2b2b;'>{st.session_state.zone_manager.score_3pt * 3}</span></strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Update Performance Stats
                fps_color = "🟢" if fps >= 10 else "🟡" if fps >= 5 else "🔴"
                stats_fps.metric("⚡ FPS", f"{fps_color} {fps:.1f}")
                avg_conf = stats.get('avg_confidence', 0)
                stats_conf.metric("🎯 Avg Confidence", f"{avg_conf:.1%}" if avg_conf > 0 else "N/A")

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
            <h2>🏀 Welcome to Basketball Detection!</h2>
            <p>To get started:</p>
            <ol>
                <li>Select <strong>Custom Ball (Trained)</strong> model from sidebar</li>
                <li>Choose Stream 2 (Fast) for real-time detection</li>
                <li>Click <strong>▶️ Start</strong> to begin detection</li>
                <li>Click <strong>⚙️ Setup Detection Zones</strong> to configure scoring zones</li>
                <li>Draw Zone 1 (above hoop) and Zone 2 (below hoop)</li>
                <li>Start shooting baskets - automatic score tracking!</li>
            </ol>
            <br>
            <p><strong>💡 Tip:</strong> Lower confidence threshold to 0.20-0.25 for better ball detection.</p>
        </div>
        """, unsafe_allow_html=True)

        # Cleanup webcam if it exists
        if st.session_state.webcam is not None:
            st.session_state.webcam.stop()
            st.session_state.webcam = None

    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0 1rem 0;">
        <p style="color: #666666; font-size: 0.9rem; margin-bottom: 0.5rem;">
            ⚡ Powered by <strong style="color: #000000;">YOLOv8</strong> •
            🎨 Built with <strong style="color: #000000;">Streamlit</strong> •
            📹 <strong style="color: #000000;">OpenCV</strong>
        </p>
        <p style="color: #999999; font-size: 0.8rem;">
            Real-time Basketball Detection & Scoring System
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
