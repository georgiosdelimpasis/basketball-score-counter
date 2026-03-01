"""
Basketball Score Counter with Roboflow Model
Uses your trained model directly from Roboflow cloud.
"""
import cv2
import json
from pathlib import Path
from roboflow import Roboflow
from roboflow_config import API_KEY, WORKSPACE, PROJECT

CAMERA_URL = "rtsp://georgedelimpasis:v!nmWiQs3Bs@192.168.1.163:554/stream2"
SETTINGS_FILE = "score_settings.json"

# Connect to Roboflow model
print("Connecting to Roboflow model...")
rf = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE).project(PROJECT)
model = project.version(1).model  # Change version number if needed
print("Model loaded!")

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
    settings = {'zone1': zone1, 'zone2': zone2}
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f)
    print(f"Settings saved")


def load_settings():
    global zone1, zone2
    if Path(SETTINGS_FILE).exists():
        with open(SETTINGS_FILE, 'r') as f:
            settings = json.load(f)
        zone1 = tuple(settings['zone1']) if settings.get('zone1') else None
        zone2 = tuple(settings['zone2']) if settings.get('zone2') else None
        print(f"Settings loaded")
        return True
    return False


def point_in_zone(x, y, zone):
    if zone is None:
        return False
    x1, y1, x2, y2 = zone
    return x1 <= x <= x2 and y1 <= y <= y2


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
        x1, y1 = start_point
        x2, y2 = x, y
        zone = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

        if current_zone == 1:
            zone1 = zone
            current_zone = 2
            print(f"Zone 1 set: {zone}")
        else:
            zone2 = zone
            print(f"Zone 2 set: {zone}")
            save_settings()


# Camera
cap = cv2.VideoCapture(CAMERA_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Failed to open camera")
    exit(1)

cv2.namedWindow('Score Counter (Roboflow)')
cv2.setMouseCallback('Score Counter (Roboflow)', mouse_callback)

for _ in range(5):
    cap.grab()

load_settings()

print("\n" + "=" * 50)
print("SCORE COUNTER - ROBOFLOW MODEL")
print("=" * 50)
print("Draw Zone 1, then Zone 2")
print("'r' = Reset zones | 's' = Reset score | 'q' = Quit")
print("=" * 50)

frame_count = 0

while True:
    cap.grab()
    ret, frame = cap.retrieve()
    if not ret:
        break

    result = frame.copy()
    ball = None

    # Only run inference every few frames for speed
    frame_count += 1
    if frame_count % 2 == 0:  # Every 2nd frame
        # Save frame temporarily for Roboflow API
        temp_path = "/tmp/frame.jpg"
        cv2.imwrite(temp_path, frame)

        try:
            # Run inference
            prediction = model.predict(temp_path, confidence=20, overlap=30).json()

            # Find ball in predictions
            for pred in prediction.get('predictions', []):
                x = int(pred['x'])
                y = int(pred['y'])
                w = int(pred['width'])
                h = int(pred['height'])
                conf = pred['confidence']

                # Use center and approximate radius
                cx, cy = x, y
                radius = max(w, h) // 2
                ball = (cx, cy, radius, conf)
                break  # Take first detection

        except Exception as e:
            pass  # Skip on error

    # Draw ball if detected
    if ball:
        cx, cy, radius, conf = ball
        cv2.circle(result, (cx, cy), radius, (0, 255, 0), 3)
        cv2.circle(result, (cx, cy), 5, (0, 255, 0), -1)
        cv2.putText(result, f"Ball {conf:.0%}", (cx - 30, cy - radius - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Check zones
        if zone1 and zone2:
            now_in_zone1 = point_in_zone(cx, cy, zone1)
            now_in_zone2 = point_in_zone(cx, cy, zone2)

            if cooldown > 0:
                cooldown -= 1

            if now_in_zone1 and not passed_zone1 and cooldown == 0:
                passed_zone1 = True
                cv2.putText(result, "ZONE 1!", (cx - 40, cy - radius - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            elif now_in_zone2 and passed_zone1 and cooldown == 0:
                score += 1
                passed_zone1 = False
                cooldown = 30
                print(f"SCORE! Total: {score}")

    # Draw zones
    if zone1:
        cv2.rectangle(result, (zone1[0], zone1[1]), (zone1[2], zone1[3]), (255, 0, 0), 2)
        cv2.putText(result, "Z1", (zone1[0] + 5, zone1[1] + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    if zone2:
        cv2.rectangle(result, (zone2[0], zone2[1]), (zone2[2], zone2[3]), (0, 0, 255), 2)
        cv2.putText(result, "Z2", (zone2[0] + 5, zone2[1] + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Drawing preview
    if drawing and start_point and end_point:
        color = (255, 0, 0) if current_zone == 1 else (0, 0, 255)
        cv2.rectangle(result, start_point, end_point, color, 2)

    # Score display
    cv2.rectangle(result, (10, 10), (200, 80), (0, 0, 0), -1)
    cv2.putText(result, f"SCORE: {score}", (20, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    # Status
    if zone1 is None:
        cv2.putText(result, "Draw ZONE 1", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    elif zone2 is None:
        cv2.putText(result, "Draw ZONE 2", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        status = "BALL DETECTED!" if ball else "Searching..."
        color = (0, 255, 0) if ball else (255, 255, 255)
        cv2.putText(result, status, (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow('Score Counter (Roboflow)', result)

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
