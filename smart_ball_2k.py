"""
Smart Ball Detector - 2K HD version.
Uses full resolution stream for better detection.
"""
import cv2
import numpy as np

# Use stream1 for FULL 2K resolution (stream2 is low-res)
CAMERA_URL = "rtsp://georgedelimpasis:v!nmWiQs3Bs@192.168.1.163:554/stream1"

def detect_purple_ball(frame):
    """Detect purple/pink basketball using color + shape analysis."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Multiple color ranges for purple/pink/magenta Lakers ball
    masks = []

    # Purple range
    masks.append(cv2.inRange(hsv, np.array([120, 25, 40]), np.array([165, 255, 255])))

    # Pink/magenta range
    masks.append(cv2.inRange(hsv, np.array([140, 20, 60]), np.array([175, 255, 255])))

    # Light purple/lavender (for highlights)
    masks.append(cv2.inRange(hsv, np.array([100, 15, 80]), np.array([145, 180, 255])))

    # Combine all masks
    combined_mask = masks[0]
    for mask in masks[1:]:
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    # Clean up mask - larger kernel for 2K
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.dilate(combined_mask, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_ball = None
    best_score = 0

    for contour in contours:
        area = cv2.contourArea(contour)

        # Larger area thresholds for 2K (4x more pixels than 720p)
        if area < 3000 or area > 800000:
            continue

        # Get bounding circle
        (cx, cy), radius = cv2.minEnclosingCircle(contour)

        # Calculate circularity
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        # Allow for partial visibility
        if circularity < 0.35:
            continue

        # Calculate fill ratio
        circle_area = np.pi * radius * radius
        fill_ratio = area / circle_area if circle_area > 0 else 0

        # Score
        score = circularity * 0.5 + fill_ratio * 0.3 + min(area / 50000, 1) * 0.2

        if score > best_score:
            best_score = score
            best_ball = {
                'center': (int(cx), int(cy)),
                'radius': int(radius),
                'confidence': min(score, 1.0),
                'area': area,
                'circularity': circularity
            }

    return best_ball, combined_mask


# Main
print("Connecting to 2K stream (stream1)...")
cap = cv2.VideoCapture(CAMERA_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Failed to open camera")
    exit(1)

# Get actual resolution
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Stream resolution: {width}x{height}")

print("=" * 60)
print("SMART BALL DETECTOR - 2K HD")
print("=" * 60)
print("Press 'm' to show color mask")
print("Press 'c' to calibrate (click on ball)")
print("Press 'q' to quit")
print("=" * 60)

show_mask = False
calibrating = False
click_hsv = None
frame_for_click = None

def mouse_callback(event, x, y, flags, param):
    global click_hsv, frame_for_click
    if event == cv2.EVENT_LBUTTONDOWN and calibrating and frame_for_click is not None:
        hsv = cv2.cvtColor(frame_for_click, cv2.COLOR_BGR2HSV)
        # Sample a small region around click for better average
        y1, y2 = max(0, y-5), min(hsv.shape[0], y+5)
        x1, x2 = max(0, x-5), min(hsv.shape[1], x+5)
        region = hsv[y1:y2, x1:x2]
        click_hsv = np.mean(region, axis=(0,1)).astype(int)
        print(f"\n>>> Ball color HSV: H={click_hsv[0]}, S={click_hsv[1]}, V={click_hsv[2]}")
        print(f">>> Add this range to detector: [{max(0,click_hsv[0]-20)}, {max(0,click_hsv[1]-40)}, {max(0,click_hsv[2]-40)}] to [{min(180,click_hsv[0]+20)}, 255, 255]")

cv2.namedWindow('Smart Ball 2K')
cv2.setMouseCallback('Smart Ball 2K', mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_for_click = frame.copy()

    # Detect ball
    ball, mask = detect_purple_ball(frame)

    result = frame.copy()

    if ball:
        cx, cy = ball['center']
        radius = ball['radius']
        conf = ball['confidence']

        # Draw thick detection circle
        cv2.circle(result, (cx, cy), radius, (0, 255, 0), 6)
        cv2.circle(result, (cx, cy), 8, (0, 255, 0), -1)

        # Label with background
        label = f"BALL {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        cv2.rectangle(result, (cx - radius, cy - radius - th - 20),
                     (cx - radius + tw + 10, cy - radius - 5), (0, 255, 0), -1)
        cv2.putText(result, label, (cx - radius + 5, cy - radius - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

        status = "BALL DETECTED!"
        color = (0, 255, 0)
    else:
        status = "Looking for purple ball..."
        color = (0, 0, 255)

    # Info overlay
    cv2.putText(result, f"2K HD | {status}",
               (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.putText(result, f"Resolution: {width}x{height}",
               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    if calibrating:
        cv2.putText(result, "CALIBRATE: Click on the ball!",
                   (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    if click_hsv is not None:
        cv2.putText(result, f"Ball HSV: H={click_hsv[0]} S={click_hsv[1]} V={click_hsv[2]}",
                   (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Resize for display if too large
    display = result
    if width > 1920:
        scale = 1920 / width
        display = cv2.resize(result, None, fx=scale, fy=scale)

    cv2.imshow('Smart Ball 2K', display)

    if show_mask:
        mask_display = mask
        if width > 1920:
            mask_display = cv2.resize(mask, None, fx=scale, fy=scale)
        cv2.imshow('Color Mask', mask_display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('m'):
        show_mask = not show_mask
        if not show_mask:
            cv2.destroyWindow('Color Mask')
    elif key == ord('c'):
        calibrating = not calibrating
        print("Calibration:", "ON - click on the ball!" if calibrating else "OFF")

cap.release()
cv2.destroyAllWindows()
