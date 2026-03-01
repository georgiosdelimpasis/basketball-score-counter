"""
Fast Purple Ball Tracker - Color-based, real-time.
Tuned for purple/magenta basketball.
"""
import cv2
import numpy as np

CAMERA_URL = "rtsp://georgedelimpasis:v!nmWiQs3Bs@192.168.1.163:554/stream2"

def find_ball(frame):
    """Find purple/magenta ball using color detection."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Purple/magenta range (your Lakers ball)
    lower = np.array([130, 30, 30])
    upper = np.array([170, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, mask

    # Find largest circular contour
    best = None
    best_score = 0

    for c in contours:
        area = cv2.contourArea(c)
        if area < 500:
            continue

        (cx, cy), radius = cv2.minEnclosingCircle(c)

        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        score = area * circularity
        if score > best_score:
            best_score = score
            best = (int(cx), int(cy), int(radius))

    return best, mask


cap = cv2.VideoCapture(CAMERA_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Flush buffer
for _ in range(5):
    cap.grab()

print("PURPLE BALL TRACKER | 'm' = mask, 'q' = quit")

show_mask = False
h_low, h_high = 130, 170  # Hue range for purple

while True:
    cap.grab()
    ret, frame = cap.retrieve()
    if not ret:
        break

    ball, mask = find_ball(frame)

    if ball:
        cx, cy, radius = ball
        cv2.circle(frame, (cx, cy), radius, (0, 255, 0), 3)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
        cv2.putText(frame, "BALL", (cx - 25, cy - radius - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Ball Tracker', frame)

    if show_mask:
        cv2.imshow('Mask', mask)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('m'):
        show_mask = not show_mask
        if not show_mask:
            cv2.destroyWindow('Mask')

cap.release()
cv2.destroyAllWindows()
