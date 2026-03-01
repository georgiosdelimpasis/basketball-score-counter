"""
Basketball Score Counter
Set 2 zones - ball passes Zone 1 → Zone 2 = SCORE!
"""
import cv2
import numpy as np
from src.hybrid_detector import HybridBallDetector

CAMERA_URL = "rtsp://georgedelimpasis:v!nmWiQs3Bs@192.168.1.163:554/stream2"

detector = HybridBallDetector()

# Zones
zone1 = None  # Entry zone (e.g., above hoop)
zone2 = None  # Exit zone (e.g., below hoop)
drawing = False
current_zone = 1
start_point = None
end_point = None

# Scoring
score = 0
passed_zone1 = False  # Ball has passed through zone1
cooldown = 0  # Prevent double counting

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
            print(f"Zone 2 set! Ready to count scores")

    elif event == cv2.EVENT_RBUTTONDOWN:
        # Right click = sample color for tracking
        hsv = detector.sample_color(frame, x, y)
        print(f"Color set: H={hsv[0]} S={hsv[1]} V={hsv[2]}")


def point_in_zone(cx, cy, zone):
    """Check if point is inside zone."""
    if zone is None:
        return False
    x1, y1, x2, y2 = zone
    return x1 <= cx <= x2 and y1 <= cy <= y2


cap = cv2.VideoCapture(CAMERA_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cv2.namedWindow('Score Counter')
cv2.setMouseCallback('Score Counter', mouse_callback)

for _ in range(5):
    cap.grab()

print("=" * 50)
print("BASKETBALL SCORE COUNTER")
print("=" * 50)
print("1. Draw ZONE 1 (entry - above hoop)")
print("2. Draw ZONE 2 (exit - below hoop)")
print("3. Right-click on ball for color tracking")
print("")
print("Ball: Zone1 → Zone2 = SCORE!")
print("")
print("Press 'r' = Reset zones")
print("Press 's' = Reset score")
print("Press 'q' = Quit")
print("=" * 50)

detector.load_ai()
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
                     (255, 0, 0), 2)  # Blue = Zone 1
        cv2.putText(result, "ZONE 1", (zone1[0], zone1[1] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    if zone2:
        cv2.rectangle(result, (zone2[0], zone2[1]), (zone2[2], zone2[3]),
                     (0, 0, 255), 2)  # Red = Zone 2
        cv2.putText(result, "ZONE 2", (zone2[0], zone2[1] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Drawing preview
    if drawing and start_point and end_point:
        color = (255, 0, 0) if current_zone == 1 else (0, 0, 255)
        cv2.rectangle(result, start_point, end_point, color, 2)

    # Detect ball
    mode = 'color' if detector.color_set else 'ai'
    detection = detector.detect(frame, prefer=mode, ai_conf=0.05)

    if detection:
        cx, cy = detection['center']
        radius = detection['radius']
        method = detection['method']

        # Draw ball
        color = (0, 255, 0) if method == 'color' else (0, 255, 255)
        cv2.circle(result, (cx, cy), radius, color, 3)
        cv2.circle(result, (cx, cy), 5, color, -1)

        # Check zones
        now_in_zone1 = point_in_zone(cx, cy, zone1)
        now_in_zone2 = point_in_zone(cx, cy, zone2)

        # Decrease cooldown
        if cooldown > 0:
            cooldown -= 1

        # Scoring logic: Zone1 → Zone2 = SCORE (strict)
        if now_in_zone1 and not passed_zone1 and cooldown == 0:
            passed_zone1 = True
            cv2.putText(result, "ZONE 1 OK!", (cx - 50, cy - radius - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        elif now_in_zone2 and passed_zone1 and cooldown == 0:
            # SCORE! Ball went Zone1 → Zone2
            score += 1
            passed_zone1 = False
            cooldown = 30  # ~1 second cooldown
            print(f"SCORE! Total: {score}")

        elif now_in_zone2 and not passed_zone1:
            # Ball in zone2 but didn't pass zone1 first - no score
            cv2.putText(result, "NO SCORE (skip Z1)", (cx - 70, cy - radius - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Reset if ball leaves both zones (new attempt)
        if not now_in_zone1 and not now_in_zone2 and passed_zone1:
            # Keep passed_zone1 true - ball is between zones
            pass

    # Score display
    cv2.rectangle(result, (10, 10), (200, 80), (0, 0, 0), -1)
    cv2.putText(result, f"SCORE: {score}", (20, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    # Instructions
    if zone1 is None:
        cv2.putText(result, "Draw ZONE 1 (entry)", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    elif zone2 is None:
        cv2.putText(result, "Draw ZONE 2 (exit)", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        tracking = "COLOR" if detector.color_set else "AI"
        cv2.putText(result, f"Tracking: {tracking} | Right-click ball for color",
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('Score Counter', result)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        zone1 = None
        zone2 = None
        current_zone = 1
        passed_zone1 = False
        cooldown = 0
        print("Zones reset")
    elif key == ord('s'):
        score = 0
        passed_zone1 = False
        cooldown = 0
        print("Score reset")

cap.release()
cv2.destroyAllWindows()
print(f"\nFinal Score: {score}")
