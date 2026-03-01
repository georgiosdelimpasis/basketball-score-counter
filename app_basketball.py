"""
Basketball Scoring System
Real-time basketball shot detection and scoring using YOLOv8.
"""
import streamlit as st
import time
import cv2
import json
import os
from src.detector import YOLODetector
from src.webcam import WebcamCapture
from src.ball_tracker import BasketballTracker
from src.color_ball_detector import ColorBallDetector
from src.circle_ball_detector import HybridBallDetector
from src.utils import (
    generate_class_colors,
    calculate_fps,
    add_fps_overlay
)
from ui.sidebar import render_sidebar
from ui.styles import inject_custom_css


# Page configuration
st.set_page_config(
    page_title="Basketball Scoring System",
    page_icon="🏀",
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

    if 'current_fps' not in st.session_state:
        st.session_state.current_fps = 0.0

    if 'class_colors' not in st.session_state:
        st.session_state.class_colors = generate_class_colors()

    if 'current_model' not in st.session_state:
        st.session_state.current_model = None

    if 'ball_tracker' not in st.session_state:
        st.session_state.ball_tracker = BasketballTracker()

    if 'hoop_zone' not in st.session_state:
        st.session_state.hoop_zone = None

    if 'setup_mode' not in st.session_state:
        st.session_state.setup_mode = False

    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False

    if 'zones' not in st.session_state:
        st.session_state.zones = {}
        # Try to load existing zones from file
        zones_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'zone_settings.json')
        if os.path.exists(zones_file_path):
            try:
                with open(zones_file_path, 'r') as f:
                    data = json.load(f)
                    st.session_state.zones = data.get('zones', {})
            except Exception as e:
                print(f"Error loading zones: {e}")
                
    if 'setup_zone_mode' not in st.session_state:
        st.session_state.setup_zone_mode = False

    if 'color_detector' not in st.session_state:
        st.session_state.color_detector = ColorBallDetector('purple')

    if 'detection_method' not in st.session_state:
        st.session_state.detection_method = 'circle'  # 'yolo', 'color', 'circle', or 'both'

    if 'ball_color' not in st.session_state:
        st.session_state.ball_color = 'purple'

    if 'circle_detector' not in st.session_state:
        st.session_state.circle_detector = HybridBallDetector(target_color='maroon')


def draw_hoop_zone(frame, hoop_zone, color=(0, 255, 0), thickness=3):
    """Draw the hoop scoring zone on the frame."""
    if hoop_zone is not None:
        try:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = hoop_zone

            # Validate and clamp coordinates
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(0, min(x2, w-1))
            y2 = max(0, min(y2, h-1))

            # Only draw if valid
            if x2 > x1 and y2 > y1:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(frame, "HOOP ZONE", (x1, max(10, y1 - 10)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        except Exception:
            pass
    return frame


def draw_target_zones(frame, zones):
    """Draw all user-defined scoring zones on the frame."""
    if not zones:
        return frame
        
    try:
        h, w = frame.shape[:2]
        for name, (x1, y1, x2, y2) in zones.items():
            # Color coding based on zone name
            if "3" in name or "Zone 2" in name:
                color = (0, 0, 255) # Red for 3-pointers
            else:
                color = (255, 0, 0) # Blue for 2-pointers
                
            x1 = max(0, min(int(x1), w-1))
            y1 = max(0, min(int(y1), h-1))
            x2 = max(0, min(int(x2), w-1))
            y2 = max(0, min(int(y2), h-1))
            
            if x2 > x1 and y2 > y1:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, name, (x1, max(10, y1 - 10)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    except Exception:
        pass
    return frame


def draw_ball_tracking(frame, ball_detections, tracker):
    """Draw basketball detection and trajectory."""
    try:
        h, w = frame.shape[:2]

        for ball in ball_detections:
            try:
                x1, y1, x2, y2 = [int(v) for v in ball['box']]

                # Validate and clamp coordinates
                x1 = max(0, min(x1, w-1))
                y1 = max(0, min(y1, h-1))
                x2 = max(0, min(x2, w-1))
                y2 = max(0, min(y2, h-1))

                # Skip invalid boxes
                if x2 <= x1 or y2 <= y1:
                    continue

                conf = ball['confidence']

                # Draw ball bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                cv2.putText(frame, f"Ball {conf:.2f}", (x1, max(10, y1 - 10)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            except Exception:
                continue

        # Draw trajectory
        trajectory = tracker.get_trajectory_line(20)
        if len(trajectory) > 1:
            for i in range(len(trajectory) - 1):
                try:
                    pt1 = trajectory[i]
                    pt2 = trajectory[i + 1]

                    # Validate points
                    if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
                        0 <= pt2[0] < w and 0 <= pt2[1] < h):
                        # Fade the line (more recent = brighter)
                        alpha = int(255 * (i + 1) / len(trajectory))
                        cv2.line(frame, pt1, pt2, (255, alpha, 0), 2)
                except Exception:
                    continue
    except Exception:
        pass

    return frame


def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()

    # Inject custom CSS
    inject_custom_css()

    # Render sidebar and get settings
    settings = render_sidebar()

    # Main content area
    st.markdown('<h1 class="main-title">🏀 Basketball Scoring System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Automatic shot detection using AI</p>', unsafe_allow_html=True)

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
            
    # Set the zones on the tracker
    if 'zones' in st.session_state:
        st.session_state.ball_tracker.set_zones(st.session_state.zones)

    st.divider()

    # Setup buttons
    if not st.session_state.webcam_active:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("🎯 Setup Hoop Zone", use_container_width=True, type="primary"):
                st.session_state.setup_mode = True
                st.rerun()
            if st.session_state.hoop_zone:
                st.success("✅ Hoop zone configured!")
            else:
                st.warning("⚠️ Hoop zone not set")
        with col2:
            if st.button("📐 Setup Target Zones (2pt/3pt)", use_container_width=True, type="primary"):
                st.session_state.setup_zone_mode = True
                st.rerun()
            if st.session_state.zones:
                st.success(f"✅ {len(st.session_state.zones)} target zones configured!")
            else:
                st.info("ℹ️ No target zones set (default to 2pt)")

        # Ball detection settings
        st.markdown("### 🏀 Ball Detection Settings")
        col1, col2 = st.columns(2)
        with col1:
            detection_method = st.selectbox(
                "Detection Method",
                options=['Circle Detection (Recommended)', 'YOLO (AI)', 'Color Detection'],
                index={'circle': 0, 'yolo': 1, 'color': 2, 'both': 0}.get(st.session_state.detection_method, 0),
                help="Circle detection works best for any ball color"
            )
            # Map selection to internal value
            method_map = {'Circle Detection (Recommended)': 'circle', 'YOLO (AI)': 'yolo', 'Color Detection': 'color'}
            st.session_state.detection_method = method_map[detection_method]

        with col2:
            if st.session_state.detection_method in ['color', 'both']:
                ball_color = st.selectbox(
                    "Ball Color",
                    options=['purple', 'orange', 'red', 'blue', 'green', 'yellow'],
                    index=['purple', 'orange', 'red', 'blue', 'green', 'yellow'].index(st.session_state.ball_color),
                    help="Select your basketball color"
                )
                if ball_color != st.session_state.ball_color:
                    st.session_state.ball_color = ball_color
                    st.session_state.color_detector = ColorBallDetector(ball_color)

        # Debug mode toggle
        st.session_state.debug_mode = st.checkbox("🐛 Debug Mode (Show ALL detections)", value=st.session_state.debug_mode,
                                                   help="Enable to see all objects detected, not just basketballs")

    # Interactive Hoop Zone Setup Mode
    if st.session_state.setup_mode:
        st.markdown("### 🎯 Interactive Hoop Zone Setup")
        st.info("📸 Adjust the sliders below to frame your basketball hoop. The green rectangle shows the scoring zone.")

        # Initialize webcam for setup
        if st.session_state.webcam is None or not st.session_state.webcam.is_active:
            st.session_state.webcam = WebcamCapture(
                camera_source=settings['camera_source'],
                resolution=settings['resolution']
            )
            if not st.session_state.webcam.start():
                st.error("❌ Failed to start camera for setup")
                st.session_state.setup_mode = False
                st.stop()

        # Capture a frame for setup
        frame = st.session_state.webcam.read()
        if frame is not None:
            h, w = frame.shape[:2]

            # Default or existing hoop zone values
            if st.session_state.hoop_zone:
                default_x1, default_y1, default_x2, default_y2 = st.session_state.hoop_zone
            else:
                default_x1, default_y1 = int(w * 0.3), int(h * 0.1)
                default_x2, default_y2 = int(w * 0.7), int(h * 0.4)

            # Sliders for hoop zone
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Top-Left Corner:**")
                hoop_x1 = st.slider("X1", 0, w, default_x1, key="x1")
                hoop_y1 = st.slider("Y1", 0, h, default_y1, key="y1")
            with col2:
                st.markdown("**Bottom-Right Corner:**")
                hoop_x2 = st.slider("X2", 0, w, default_x2, key="x2")
                hoop_y2 = st.slider("Y2", 0, h, default_y2, key="y2")

            # Draw preview
            preview_frame = frame.copy()
            preview_frame = draw_hoop_zone(preview_frame, (hoop_x1, hoop_y1, hoop_x2, hoop_y2), color=(0, 255, 0), thickness=3)

            # Add instructions on frame
            cv2.putText(preview_frame, "Adjust sliders to frame the hoop", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            st.image(preview_frame, channels="BGR", use_column_width=True)

            # Control buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("✅ Save & Continue", use_container_width=True, type="primary"):
                    st.session_state.hoop_zone = (hoop_x1, hoop_y1, hoop_x2, hoop_y2)
                    st.session_state.ball_tracker.set_hoop_zone(hoop_x1, hoop_y1, hoop_x2, hoop_y2)
                    st.session_state.setup_mode = False
                    # Stop webcam
                    st.session_state.webcam.stop()
                    st.session_state.webcam = None
                    st.success("✅ Hoop zone saved!")
                    time.sleep(1)
                    st.rerun()
            with col2:
                if st.button("🔄 Reset to Default", use_container_width=True):
                    st.rerun()
            with col3:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state.setup_mode = False
                    # Stop webcam
                    if st.session_state.webcam:
                        st.session_state.webcam.stop()
                        st.session_state.webcam = None
                    st.rerun()
        else:
            st.error("❌ Could not capture frame from camera")
            if st.button("« Back"):
                st.session_state.setup_mode = False
                st.rerun()

        st.stop()  # Stop processing here during setup mode

    # Interactive Target Zone Setup Mode
    if st.session_state.get('setup_zone_mode', False):
        st.markdown("### 📐 Interactive Target Zone Setup")
        st.info("📸 Define zones on the court. Shots originating from '3-Point' zones are worth 3 points, others are worth 2.")

        # Initialize webcam for setup
        if st.session_state.webcam is None or not st.session_state.webcam.is_active:
            st.session_state.webcam = WebcamCapture(
                camera_source=settings['camera_source'],
                resolution=settings['resolution']
            )
            if not st.session_state.webcam.start():
                st.error("❌ Failed to start camera for setup")
                st.session_state.setup_zone_mode = False
                st.stop()

        # Capture a frame for setup
        frame = st.session_state.webcam.read()
        if frame is not None:
            h, w = frame.shape[:2]

            # List existing zones
            if st.session_state.zones:
                st.markdown("**Current Zones:**")
                for name, coords in list(st.session_state.zones.items()):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.write(f"- {name}: {coords}")
                    with col2:
                        if st.button("🗑️ Delete", key=f"del_{name}"):
                            del st.session_state.zones[name]
                            # Save to file
                            try:
                                zones_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'zone_settings.json')
                                with open(zones_file_path, 'w') as f:
                                    json.dump({"zones": st.session_state.zones}, f, indent=2)
                            except Exception as e:
                                st.error(f"Error saving zones: {e}")
                            st.rerun()
                            
            st.divider()
            st.markdown("**Add New Zone:**")
            
            # Form for new zone
            col1, col2 = st.columns(2)
            with col1:
                zone_name = st.text_input("Zone Name", "Zone " + str(len(st.session_state.zones) + 1))
                zone_type = st.selectbox("Points", options=[2, 3], index=1 if "3" in str(len(st.session_state.zones) + 1) else 0)
                if zone_type == 3 and "3" not in zone_name:
                    zone_name = f"{zone_name} (3P)"
            
            default_x1, default_y1 = int(w * 0.1), int(h * 0.1)
            default_x2, default_y2 = int(w * 0.4), int(h * 0.8)

            # Sliders for target zone
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Top-Left:**")
                zx1 = st.slider("X1", 0, w, default_x1, key="zx1")
                zy1 = st.slider("Y1", 0, h, default_y1, key="zy1")
            with col2:
                st.markdown("**Bottom-Right:**")
                zx2 = st.slider("X2", 0, w, default_x2, key="zx2")
                zy2 = st.slider("Y2", 0, h, default_y2, key="zy2")

            # Draw preview
            preview_frame = frame.copy()
            if st.session_state.hoop_zone:
                preview_frame = draw_hoop_zone(preview_frame, st.session_state.hoop_zone, color=(0, 255, 0), thickness=2)
            preview_frame = draw_target_zones(preview_frame, st.session_state.zones)
            
            # Draw the current new zone being edited in yellow
            cv2.rectangle(preview_frame, (zx1, zy1), (zx2, zy2), (0, 255, 255), 3)
            cv2.putText(preview_frame, zone_name + " (New)", (zx1, max(10, zy1 - 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            st.image(preview_frame, channels="BGR", use_column_width=True)

            # Control buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("➕ Add Zone", use_container_width=True, type="primary"):
                    st.session_state.zones[zone_name] = [zx1, zy1, zx2, zy2]
                    st.session_state.ball_tracker.set_zones(st.session_state.zones)
                    
                    # Save to file
                    try:
                        zones_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'zone_settings.json')
                        with open(zones_file_path, 'w') as f:
                            json.dump({"zones": st.session_state.zones}, f, indent=2)
                        st.success("✅ Zone saved!")
                    except Exception as e:
                        st.error(f"Error saving zones: {e}")
                    time.sleep(1)
                    st.rerun()
            with col2:
                if st.button("❌ Done", use_container_width=True):
                    st.session_state.setup_zone_mode = False
                    # Stop webcam
                    if st.session_state.webcam:
                        st.session_state.webcam.stop()
                        st.session_state.webcam = None
                    st.rerun()
            with col3:
                pass
        else:
            st.error("❌ Could not capture frame from camera")
            if st.button("« Back"):
                st.session_state.setup_zone_mode = False
                st.rerun()

        st.stop()  # Stop processing here during setup mode

    # Webcam processing
    if settings['webcam_active']:
        # Check if hoop zone is set
        if st.session_state.hoop_zone is None:
            st.warning("⚠️ Please set up the hoop zone first!")

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
            st.markdown("### 🏀 Scoring Stats")
            score_display = st.empty()
            score_breakdown = st.empty()
            stats_fps = st.empty()
            stats_ball = st.empty()
            st.markdown("---")
            if st.button("🔄 Reset Score"):
                st.session_state.ball_tracker.reset_score()
                st.success("Score reset!")

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

                # Run detection based on selected method
                ball_detections = []

                if st.session_state.detection_method in ['yolo', 'both']:
                    # YOLO detection
                    detections = st.session_state.detector.detect(
                        frame,
                        conf_threshold=settings['conf_threshold'],
                        iou_threshold=settings['iou_threshold'],
                        image_size=settings['image_size']
                    )
                    # Filter to only show sports balls (basketball)
                    yolo_balls = [d for d in detections if d['class_name'] == 'sports ball']
                    ball_detections.extend(yolo_balls)
                else:
                    detections = []

                if st.session_state.detection_method == 'circle':
                    # Circle-based detection (shape-based, works for any color)
                    circle_balls = st.session_state.circle_detector.detect(frame, min_radius=20, max_radius=400)
                    ball_detections.extend(circle_balls)

                if st.session_state.detection_method in ['color', 'both']:
                    # Color-based detection (very lenient for close-up balls)
                    color_balls = st.session_state.color_detector.detect(frame, min_radius=5, max_radius=800)
                    ball_detections.extend(color_balls)

                # Remove duplicates if using both methods (take highest confidence)
                if st.session_state.detection_method == 'both' and len(ball_detections) > 1:
                    # Simple deduplication: keep ball with highest confidence
                    ball_detections = [max(ball_detections, key=lambda x: x['confidence'])]

                # Update tracker and check for scoring
                scored, points_awarded = st.session_state.ball_tracker.update(ball_detections)
                if scored:
                    if points_awarded == 3:
                        status_placeholder.success("🎯 3-POINTER!", icon="🔥")
                    else:
                        status_placeholder.success("🎉 SCORE!", icon="🏀")

                # Draw hoop zone
                annotated_frame = frame.copy()
                if st.session_state.hoop_zone:
                    annotated_frame = draw_hoop_zone(annotated_frame, st.session_state.hoop_zone)
                    
                # Draw target zones
                if st.session_state.zones:
                    annotated_frame = draw_target_zones(annotated_frame, st.session_state.zones)

                # Debug mode: show ALL detections (limit to first 50 to prevent crashes)
                if st.session_state.debug_mode:
                    h, w = annotated_frame.shape[:2]

                    # Show YOLO detections
                    for det in detections[:50]:  # Limit to 50 to prevent overload
                        try:
                            x1, y1, x2, y2 = [int(v) for v in det['box']]

                            # Validate coordinates to prevent crashes
                            x1 = max(0, min(x1, w-1))
                            y1 = max(0, min(y1, h-1))
                            x2 = max(0, min(x2, w-1))
                            y2 = max(0, min(y2, h-1))

                            # Skip invalid boxes
                            if x2 <= x1 or y2 <= y1:
                                continue

                            conf = det['confidence']
                            label = f"{det['class_name']} {conf:.2f}"
                            color = (0, 255, 255) if det['class_name'] == 'sports ball' else (128, 128, 128)
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(annotated_frame, label, (x1, max(10, y1 - 10)),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        except Exception as e:
                            # Skip problematic detections silently
                            continue

                    # Show circle detections (green circles)
                    if st.session_state.detection_method == 'circle':
                        try:
                            circle_balls = st.session_state.circle_detector.detect(frame, min_radius=20, max_radius=400)
                            for det in circle_balls[:5]:
                                try:
                                    cx, cy = det['center']
                                    radius = det['radius']
                                    conf = det['confidence']

                                    # Validate coordinates
                                    if 0 <= cx < w and 0 <= cy < h:
                                        cv2.circle(annotated_frame, (cx, cy), radius, (0, 255, 0), 3)
                                        cv2.circle(annotated_frame, (cx, cy), 5, (0, 255, 0), -1)
                                        label = f"Ball R:{radius} C:{conf:.2f}"
                                        cv2.putText(annotated_frame, label, (cx - radius, max(10, cy - radius - 10)),
                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                except Exception:
                                    continue
                        except Exception:
                            pass

                    # Show color detections in different color
                    if st.session_state.detection_method in ['color', 'both']:
                        try:
                            color_balls = st.session_state.color_detector.detect(frame, min_radius=5, max_radius=800)
                            for det in color_balls[:10]:  # Limit color detections
                                try:
                                    x1, y1, x2, y2 = [int(v) for v in det['box']]

                                    # Validate coordinates
                                    x1 = max(0, min(x1, w-1))
                                    y1 = max(0, min(y1, h-1))
                                    x2 = max(0, min(x2, w-1))
                                    y2 = max(0, min(y2, h-1))

                                    # Skip invalid boxes
                                    if x2 <= x1 or y2 <= y1:
                                        continue

                                    conf = det['confidence']
                                    label = f"{det['class_name']} {conf:.2f}"
                                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)  # Magenta
                                    cv2.putText(annotated_frame, label, (x1, max(10, y1 - 10)),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                                except Exception:
                                    continue
                        except Exception:
                            # If color detection fails entirely, continue without it
                            pass

                # Draw ball tracking (only if not in debug mode)
                elif ball_detections:
                    annotated_frame = draw_ball_tracking(annotated_frame, ball_detections, st.session_state.ball_tracker)

                # Calculate and add FPS
                frame_count += 1
                fps = calculate_fps(start_time, frame_count)
                st.session_state.current_fps = fps
                annotated_frame = add_fps_overlay(annotated_frame, fps)

                # Update real-time statistics display
                current_score = st.session_state.ball_tracker.total_score
                score_display.metric("🎯 TOTAL SCORE", current_score, help="Total points made")
                
                # Show breakdown
                score_breakdown.markdown(f"""
                <div style='background-color: #212529; padding: 10px; border-radius: 5px; margin-bottom: 10px'>
                    <div style='display: flex; justify-content: space-between;'>
                        <span>2-Pointers:</span> <strong>{st.session_state.ball_tracker.score_2pt} x 2 = {st.session_state.ball_tracker.score_2pt * 2}</strong>
                    </div>
                    <div style='display: flex; justify-content: space-between;'>
                        <span>3-Pointers:</span> <strong>{st.session_state.ball_tracker.score_3pt} x 3 = {st.session_state.ball_tracker.score_3pt * 3}</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                fps_color = "🟢" if fps >= 10 else "🟡" if fps >= 5 else "🔴"
                stats_fps.metric("⚡ FPS", f"{fps_color} {fps:.1f}")

                ball_detected = "Yes ✅" if len(ball_detections) > 0 else "No ❌"
                stats_ball.metric("🏀 Ball Detected", ball_detected)

                # Debug info
                if st.session_state.debug_mode:
                    st.markdown("**🐛 Debug Info:**")
                    st.write(f"Total objects detected: {len(detections)}")
                    if detections:
                        st.write("Detected classes:")
                        class_counts = {}
                        for d in detections:
                            cls = d['class_name']
                            class_counts[cls] = class_counts.get(cls, 0) + 1
                        for cls, count in class_counts.items():
                            st.write(f"• {cls}: {count}")

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
            <h2>🏀 Welcome to Basketball Scoring System!</h2>
            <p>To get started:</p>
            <ol>
                <li>Set up the hoop zone coordinates above</li>
                <li>Choose YOLOv8n model (fastest for real-time)</li>
                <li>Select your camera source</li>
                <li>Click <strong>▶️ Start</strong> to begin tracking</li>
                <li>Shoot some hoops and watch the score increase!</li>
            </ol>
            <br>
            <p><strong>💡 Tips:</strong></p>
            <ul>
                <li>Position camera with clear view of the hoop</li>
                <li>Ensure good lighting for best detection</li>
                <li>Orange basketballs work best for detection</li>
                <li>Keep camera steady for accurate tracking</li>
            </ul>
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
        Basketball Scoring System | Built with YOLOv8, Streamlit, and OpenCV
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
