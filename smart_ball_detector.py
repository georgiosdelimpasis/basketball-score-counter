"""
Smart Ball Detector - Robust detection for purple/pink basketball.
Uses color segmentation + contour analysis + tracking.
"""
import cv2
import numpy as np

CAMERA_URL = "rtsp://georgedelimpasis:v!nmWiQs3Bs@192.168.1.163:554/stream2"

def detect_purple_ball(frame):
    """Detect purple/pink basketball using color + shape analysis."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Multiple color ranges for purple/pink/magenta ball
    masks = []

    # Purple range (the main color of your Lakers ball)
    masks.append(cv2.inRange(hsv, np.array([120, 30, 50]), np.array([160, 255, 255])))

    # Pink/magenta range
    masks.append(cv2.inRange(hsv, np.array([140, 30, 50]), np.array([175, 255, 255])))

    # Light purple/lavender
    masks.append(cv2.inRange(hsv, np.array([100, 20, 100]), np.array([140, 150, 255])))

    # Combine all masks
    combined_mask = masks[0]
    for mask in masks[1:]:
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    # Clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.dilate(combined_mask, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_ball = None
    best_score = 0

    for contour in contours:
        area = cv2.contourArea(contour)

        # Ball should be reasonably sized (adjust based on distance)
        if area < 1000 or area > 200000:
            continue

        # Get bounding circle
        (cx, cy), radius = cv2.minEnclosingCircle(contour)

        # Calculate circularity
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        # Ball should be somewhat circular (but allow for occlusion/angle)
        if circularity < 0.4:
            continue

        # Calculate how well the contour fills its bounding circle
        circle_area = np.pi * radius * radius
        fill_ratio = area / circle_area if circle_area > 0 else 0

        # Score based on circularity, size, and fill
        score = circularity * 0.4 + fill_ratio * 0.4 + min(area / 10000, 1) * 0.2

        if score > best_score:
            best_score = score
            best_ball = {
                'center': (int(cx), int(cy)),
                'radius': int(radius),
                'confidence': score,
                'area': area,
                'circularity': circularity
            }

    return best_ball, combined_mask


# Main
cap = cv2.VideoCapture(CAMERA_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Failed to open camera")
    exit(1)

print("=" * 60)
print("SMART PURPLE BALL DETECTOR")
print("=" * 60)
print("Press 'm' to show color mask")
print("Press 'c' to calibrate (click on ball)")
print("Press 'q' to quit")
print("=" * 60)

show_mask = False
calibrating = False
click_hsv = None

def mouse_callback(event, x, y, flags, param):
    global click_hsv, frame_for_click
    if event == cv2.EVENT_LBUTTONDOWN and calibrating:
        hsv = cv2.cvtColor(frame_for_click, cv2.COLOR_BGR2HSV)
        click_hsv = hsv[y, x]
        print(f"\nClicked HSV: H={click_hsv[0]}, S={click_hsv[1]}, V={click_hsv[2]}")
        print(f"Suggested range: H=[{max(0,click_hsv[0]-15)}-{min(180,click_hsv[0]+15)}]")

cv2.namedWindow('Smart Ball Detection')
cv2.setMouseCallback('Smart Ball Detection', mouse_callback)

frame_for_click = None

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

        # Draw detection
        cv2.circle(result, (cx, cy), radius, (0, 255, 0), 4)
        cv2.circle(result, (cx, cy), 5, (0, 255, 0), -1)

        # Label
        label = f"BALL {conf:.0%}"
        cv2.putText(result, label, (cx - radius, cy - radius - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        status = "BALL DETECTED!"
        color = (0, 255, 0)
    else:
        status = "Looking for purple ball..."
        color = (0, 0, 255)

    # Info overlay
    cv2.putText(result, f"Smart Detector | {status}",
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    if calibrating:
        cv2.putText(result, "CALIBRATE: Click on the ball!",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if click_hsv is not None:
        cv2.putText(result, f"Last HSV: H={click_hsv[0]} S={click_hsv[1]} V={click_hsv[2]}",
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('Smart Ball Detection', result)

    if show_mask:
        cv2.imshow('Color Mask (white = detected)', mask)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('m'):
        show_mask = not show_mask
        if not show_mask:
            cv2.destroyWindow('Color Mask (white = detected)')
    elif key == ord('c'):
        calibrating = not calibrating
        print("Calibration mode:", "ON - click on the ball" if calibrating else "OFF")

cap.release()
cv2.destroyAllWindows()
