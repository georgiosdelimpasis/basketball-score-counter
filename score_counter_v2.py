"""
Basketball Score Counter v2
- AI ball detection (YOLO)
- Saves zones settings
"""
import cv2
import numpy as np
import json
from pathlib import Path

CAMERA_URL = "rtsp://georgedelimpasis:v!nmWiQs3Bs@192.168.1.163:554/stream2"
SETTINGS_FILE = "score_settings.json"

# Motion detector
print("Initializing motion detector...")
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=50,
    varThreshold=40,
    detectShadows=False
)
min_ball_radius = 20
max_ball_radius = 150

# Zones
zone1 = None
zone2 = None
drawing = False
current_zone = 1
start_point = None
end_point = None

# Scoring
score = 0
passed_zone1 = False
cooldown = 0



def save_settings():
    """Save zones to file."""
    settings = {
        'zone1': zone1,
        'zone2': zone2,
        'min_radius': min_ball_radius,
        'max_radius': max_ball_radius
    }
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f)
    print(f"Settings saved to {SETTINGS_FILE}")


def load_settings():
    """Load zones from file."""
    global zone1, zone2, min_ball_radius, max_ball_radius
    if Path(SETTINGS_FILE).exists():
        with open(SETTINGS_FILE, 'r') as f:
            settings = json.load(f)
        zone1 = tuple(settings['zone1']) if settings.get('zone1') else None
        zone2 = tuple(settings['zone2']) if settings.get('zone2') else None
        min_ball_radius = settings.get('min_radius', 20)
        max_ball_radius = settings.get('max_radius', 150)
        print(f"Settings loaded from {SETTINGS_FILE}")
        if zone1:
            print(f"Zone 1: {zone1}")
        if zone2:
            print(f"Zone 2: {zone2}")
        return True
    return False


def detect_ball(frame):
    """Detect moving circular object."""
    # Apply background subtraction
    fg_mask = bg_subtractor.apply(frame)

    # Clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_ball = None
    best_score = 0

    for c in contours:
        area = cv2.contourArea(c)
        if area < 500:
            continue

        # Get enclosing circle
        (cx, cy), radius = cv2.minEnclosingCircle(c)
        cx, cy, radius = int(cx), int(cy), int(radius)

        # Check size
        if radius < min_ball_radius or radius > max_ball_radius:
            continue

        # Check circularity
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        if circularity < 0.4:
            continue

        # Score
        score_val = circularity * area
        if score_val > best_score:
            best_score = score_val
            best_ball = (cx, cy, radius)

    return best_ball


def mouse_callback(event, x, y, flags, param):
    global zone1, zone2, drawing, start_point, end_point, current_zone

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        end_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        x1, y1 = min(start_point[0], end_point[0]), min(start_point[1], end_point[1])
        x2, y2 = max(start_point[0], end_point[0]), max(start_point[1], end_point[1])

        if current_zone == 1:
            zone1 = (x1, y1, x2, y2)
            current_zone = 2
            print(f"Zone 1 set! Now draw Zone 2")
        else:
            zone2 = (x1, y1, x2, y2)
            current_zone = 1
            print(f"Zone 2 set!")
        save_settings()


def point_in_zone(cx, cy, zone):
    if zone is None:
        return False
    x1, y1, x2, y2 = zone
    return x1 <= cx <= x2 and y1 <= cy <= y2


# Main
cap = cv2.VideoCapture(CAMERA_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cv2.namedWindow('Score Counter')
cv2.setMouseCallback('Score Counter', mouse_callback)

for _ in range(5):
    cap.grab()

# Load saved settings
load_settings()

print("\n" + "=" * 50)
print("BASKETBALL SCORE COUNTER v2 (MOTION)")
print("=" * 50)
print("Draw Zone 1, then Zone 2")
print("MOVE THE BALL to detect it!")
print("'r' = Reset zones")
print("'s' = Reset score")
print("'q' = Quit")
print("=" * 50)

frame = None

while True:
    cap.grab()
    ret, frame = cap.retrieve()
    if not ret:
        break

    result = frame.copy()

    # Draw zones
    if zone1:
        cv2.rectangle(result, (zone1[0], zone1[1]), (zone1[2], zone1[3]),
                     (255, 0, 0), 2)
        cv2.putText(result, "ZONE 1", (zone1[0], zone1[1] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    if zone2:
        cv2.rectangle(result, (zone2[0], zone2[1]), (zone2[2], zone2[3]),
                     (0, 0, 255), 2)
        cv2.putText(result, "ZONE 2", (zone2[0], zone2[1] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Drawing preview
    if drawing and start_point and end_point:
        color = (255, 0, 0) if current_zone == 1 else (0, 0, 255)
        cv2.rectangle(result, start_point, end_point, color, 2)

    # Detect ball
    ball = detect_ball(frame)

    if ball:
        cx, cy, radius = ball
        cv2.circle(result, (cx, cy), radius, (0, 255, 0), 3)
        cv2.circle(result, (cx, cy), 5, (0, 255, 0), -1)

        # Check zones
        now_in_zone1 = point_in_zone(cx, cy, zone1)
        now_in_zone2 = point_in_zone(cx, cy, zone2)

        if cooldown > 0:
            cooldown -= 1

        # Scoring logic
        if now_in_zone1 and not passed_zone1 and cooldown == 0:
            passed_zone1 = True
            cv2.putText(result, "ZONE 1 OK!", (cx - 50, cy - radius - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        elif now_in_zone2 and passed_zone1 and cooldown == 0:
            score += 1
            passed_zone1 = False
            cooldown = 30
            print(f"SCORE! Total: {score}")

        elif now_in_zone2 and not passed_zone1:
            cv2.putText(result, "NO (need Z1)", (cx - 50, cy - radius - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Score display
    cv2.rectangle(result, (10, 10), (200, 80), (0, 0, 0), -1)
    cv2.putText(result, f"SCORE: {score}", (20, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    # Status bar
    if zone1 is None:
        cv2.putText(result, "Draw ZONE 1 (entry)", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    elif zone2 is None:
        cv2.putText(result, "Draw ZONE 2 (exit)", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        status = "BALL MOVING!" if ball else "Move the ball..."
        color = (0, 255, 0) if ball else (255, 255, 255)
        cv2.putText(result, f"MOTION: {status}", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow('Score Counter', result)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        zone1 = None
        zone2 = None
        current_zone = 1
        passed_zone1 = False
        save_settings()
        print("Zones reset")
    elif key == ord('s'):
        score = 0
        passed_zone1 = False
        cooldown = 0
        print("Score reset")

cap.release()
cv2.destroyAllWindows()
print(f"\nFinal Score: {score}")
