"""
Zone setup UI component for drawing detection zones.
"""
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
from PIL import Image
from typing import Optional


def render_zone_setup(webcam, zone_manager) -> bool:
    """
    Render zone setup interface.

    Args:
        webcam: Active webcam capture object
        zone_manager: ZoneManager instance

    Returns:
        True if zones were updated
    """
    st.markdown("""
    <div class="zone-card">
        <h3>🎯 Zone Setup</h3>
        <p style="color: #333333; line-height: 1.8;">
            Draw detection zones on the video feed:<br>
            🔴 <strong>Zone 1:</strong> Upper zone (ball enters hoop)<br>
            🔵 <strong>Zone 2:</strong> Lower zone (ball exits hoop)<br>
            🏀 When ball passes <strong>Zone 1 → Zone 2</strong>, it counts as a score!
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")  # Spacer

    # Capture current frame
    if webcam is not None and webcam.is_active:
        frame = webcam.read()
        if frame is not None:
            # Convert BGR to RGB for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]

            # Display current zones if they exist
            if zone_manager.zones:
                st.markdown("**Current Zones:**")
                for zone_name, (x1, y1, x2, y2) in zone_manager.zones.items():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.text(f"{zone_name}: ({x1}, {y1}) → ({x2}, {y2})")
                    with col2:
                        if st.button("❌", key=f"del_{zone_name}"):
                            zone_manager.remove_zone(zone_name)
                            st.rerun()

            st.divider()

            # Zone selection
            zone_to_draw = st.selectbox(
                "Select Zone to Draw",
                ["Zone 1", "Zone 2", "Zone 3", "Zone 4"],
                help="Draw rectangles to define detection zones"
            )

            st.markdown("**Instructions:**")
            st.markdown("""
            1. Select a zone from the dropdown above
            2. Draw a rectangle on the image below
            3. Click 'Save Zones' to apply
            4. Repeat for additional zones
            """)

            # Drawing canvas
            canvas_result = st_canvas(
                fill_color="rgba(255, 0, 0, 0.3)",
                stroke_width=2,
                stroke_color="#FF0000" if zone_to_draw == "Zone 1" else "#0000FF",
                background_image=Image.fromarray(frame_rgb),
                update_streamlit=True,
                height=h,
                width=w,
                drawing_mode="rect",
                key="canvas",
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("💾 Save Zones", use_container_width=True):
                    if canvas_result.json_data is not None:
                        objects = canvas_result.json_data.get("objects", [])
                        if objects:
                            # Get the last drawn rectangle
                            rect = objects[-1]
                            x1 = int(rect["left"])
                            y1 = int(rect["top"])
                            x2 = int(rect["left"] + rect["width"])
                            y2 = int(rect["top"] + rect["height"])

                            zone_manager.add_zone(zone_to_draw, x1, y1, x2, y2)
                            st.success(f"✅ {zone_to_draw} saved!")
                            return True
                        else:
                            st.warning("⚠️ Draw a rectangle first")

            with col2:
                if st.button("🗑️ Clear All", use_container_width=True):
                    zone_manager.clear_zones()
                    st.success("✅ All zones cleared")
                    st.rerun()

            with col3:
                if st.button("✖️ Close", use_container_width=True):
                    st.session_state.setup_zones = False
                    st.rerun()

        else:
            st.warning("⚠️ No frame available. Start the webcam first.")
    else:
        st.warning("⚠️ Please start the webcam before setting up zones.")

    return False
