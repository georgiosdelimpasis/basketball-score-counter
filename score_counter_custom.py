"""
Basketball Score Counter with Custom YOLO Model
Uses your locally trained ball detector.
"""
import cv2
import json
from pathlib import Path
from ultralytics import YOLO

CAMERA_URL = "rtsp://georgedelimpasis:v!nmWiQs3Bs@192.168.1.163:554/stream2"
SETTINGS_FILE = "score_settings.json"

# Find trained model
model_paths = [
    "runs/detect/my_ball/weights/best.pt",
    "runs/detect/my_ball2/weights/best.pt",
    "runs/detect/my_ball3/weights/best.pt",
    "best.pt",
    "my_ball.pt"
]

model_path = None
for p in model_paths:
    if Path(p).exists():
        model_path = p
        break

if model_path is None:
    print("ERROR: No trained model found!")
    print("Run train_my_ball.py first to train your model.")
    exit(1)

print(f"Loading model: {model_path}")
model = YOLO(model_path)
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
    print("Settings saved")


def load_settings():
    global zone1, zone2
    if Path(SETTINGS_FILE).exists():
        with open(SETTINGS_FILE, 'r') as f:
            settings = json.load(f)
        zone1 = tuple(settings['zone1']) if settings.get('zone1') else None
        zone2 = tuple(settings['zone2']) if settings.get('zone2') else None
        print("Settings loaded")
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
            print(f"Zone 1 set")
        else:
            zone2 = zone
            print(f"Zone 2 set")
            save_settings()


# Camera
cap = cv2.VideoCapture(CAMERA_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Failed to open camera")
    exit(1)

cv2.namedWindow('Score Counter (Custom Model)')
cv2.setMouseCallback('Score Counter (Custom Model)', mouse_callback)

for _ in range(5):
    cap.grab()

load_settings()

print("\n" + "=" * 50)
print("SCORE COUNTER - CUSTOM TRAINED MODEL")
print("=" * 50)
print(f"Model: {model_path}")
print("Draw Zone 1, then Zone 2")
print("'r' = Reset zones | 's' = Reset score | 'q' = Quit")
print("=" * 50)

while True:
    cap.grab()
    ret, frame = cap.retrieve()
    if not ret:
        break

    result = frame.copy()
    ball = None

    # Run YOLO detection
    results = model(frame, conf=0.25, verbose=False)

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            # Get center and radius
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            radius = max(x2 - x1, y2 - y1) // 2

            ball = (cx, cy, radius, conf)

            # Draw detection
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(result, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(result, f"Ball {conf:.0%}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            break  # Take first detection

    # Check zones and scoring
    if ball and zone1 and zone2:
        cx, cy, radius, conf = ball
        now_in_zone1 = point_in_zone(cx, cy, zone1)
        now_in_zone2 = point_in_zone(cx, cy, zone2)

        if cooldown > 0:
            cooldown -= 1

        if now_in_zone1 and not passed_zone1 and cooldown == 0:
            passed_zone1 = True
            cv2.putText(result, "ZONE 1!", (cx - 40, cy - radius - 20),
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
        status = "BALL DETECTED!" if ball else "Looking..."
        color = (0, 255, 0) if ball else (255, 255, 255)
        cv2.putText(result, status, (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow('Score Counter (Custom Model)', result)

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
