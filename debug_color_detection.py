"""
Color Detection Debug Tool
Helps calibrate HSV color ranges for ball detection.
"""
import cv2
import numpy as np

# RTSP camera URL
CAMERA_URL = "rtsp://georgedelimpasis:v!nmWiQs3Bs@192.168.1.163:554/stream2"

def nothing(x):
    """Callback for trackbars."""
    pass

def main():
    print("🎨 Color Detection Calibration Tool")
    print("=" * 50)

    # Open camera
    cap = cv2.VideoCapture(CAMERA_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("❌ Failed to open camera")
        return

    print("✅ Camera connected")
    print("\nInstructions:")
    print("- Adjust the HSV sliders to isolate your purple ball")
    print("- The MASK window shows what's being detected (white = detected)")
    print("- Press 'q' to quit")
    print("- Press 's' to save current HSV values")
    print("\nCurrent purple range: H(120-150), S(50-255), V(50-255)")

    # Create window and trackbars
    cv2.namedWindow('Controls')
    cv2.namedWindow('Frame')
    cv2.namedWindow('Mask')

    # HSV range trackbars - start with purple range
    cv2.createTrackbar('H Min', 'Controls', 120, 179, nothing)
    cv2.createTrackbar('H Max', 'Controls', 150, 179, nothing)
    cv2.createTrackbar('S Min', 'Controls', 50, 255, nothing)
    cv2.createTrackbar('S Max', 'Controls', 255, 255, nothing)
    cv2.createTrackbar('V Min', 'Controls', 50, 255, nothing)
    cv2.createTrackbar('V Max', 'Controls', 255, 255, nothing)

    # Detection parameters
    cv2.createTrackbar('Min Radius', 'Controls', 10, 100, nothing)
    cv2.createTrackbar('Max Radius', 'Controls', 200, 500, nothing)
    cv2.createTrackbar('Circularity*10', 'Controls', 6, 10, nothing)  # 0.6 = 6/10

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to read frame")
            break

        # Get trackbar values
        h_min = cv2.getTrackbarPos('H Min', 'Controls')
        h_max = cv2.getTrackbarPos('H Max', 'Controls')
        s_min = cv2.getTrackbarPos('S Min', 'Controls')
        s_max = cv2.getTrackbarPos('S Max', 'Controls')
        v_min = cv2.getTrackbarPos('V Min', 'Controls')
        v_max = cv2.getTrackbarPos('V Max', 'Controls')

        min_radius = cv2.getTrackbarPos('Min Radius', 'Controls')
        max_radius = cv2.getTrackbarPos('Max Radius', 'Controls')
        circularity_threshold = cv2.getTrackbarPos('Circularity*10', 'Controls') / 10.0

        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create mask
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(hsv, lower, upper)

        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw detections on frame
        result = frame.copy()
        detection_count = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < (min_radius * min_radius * 3.14):
                continue

            (x, y), radius = cv2.minEnclosingCircle(contour)

            if min_radius <= radius <= max_radius:
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue

                circularity = 4 * np.pi * area / (perimeter * perimeter)

                # Draw all contours in gray
                cv2.circle(result, (int(x), int(y)), int(radius), (128, 128, 128), 2)

                # Highlight circular objects in green
                if circularity > circularity_threshold:
                    cv2.circle(result, (int(x), int(y)), int(radius), (0, 255, 0), 3)
                    cv2.putText(result, f"C:{circularity:.2f} R:{int(radius)}",
                              (int(x - radius), int(y - radius - 10)),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    detection_count += 1
                else:
                    cv2.putText(result, f"C:{circularity:.2f}",
                              (int(x - radius), int(y - radius - 10)),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)

        # Add info text
        info_text = f"Detections: {detection_count} | Contours: {len(contours)}"
        cv2.putText(result, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        hsv_text = f"HSV: [{h_min},{s_min},{v_min}] - [{h_max},{s_max},{v_max}]"
        cv2.putText(result, hsv_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Show windows
        cv2.imshow('Frame', result)
        cv2.imshow('Mask', mask)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            print(f"\n✅ Current HSV values:")
            print(f"   'purple': ([{h_min}, {s_min}, {v_min}], [{h_max}, {s_max}, {v_max}])")
            print(f"   Min Radius: {min_radius}")
            print(f"   Max Radius: {max_radius}")
            print(f"   Circularity: {circularity_threshold}")
            print("\nCopy this to src/color_ball_detector.py if it works well!")

    cap.release()
    cv2.destroyAllWindows()
    print("\n👋 Calibration tool closed")

if __name__ == "__main__":
    main()
