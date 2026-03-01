"""
Use your Roboflow-trained model for ball detection.
Put your best.pt file in this folder, then run this script.
"""
import cv2
from ultralytics import YOLO
from pathlib import Path

CAMERA_URL = "rtsp://georgedelimpasis:v!nmWiQs3Bs@192.168.1.163:554/stream2"

# Find model file
model_path = None
for path in ["best.pt", "my_ball.pt", "ball_model.pt", "runs/detect/my_ball/weights/best.pt"]:
    if Path(path).exists():
        model_path = path
        break

if model_path is None:
    print("=" * 60)
    print("ERROR: No model file found!")
    print("=" * 60)
    print("\nPut your trained model (best.pt) in this folder.")
    print("Or rename it to: my_ball.pt or ball_model.pt")
    exit(1)

print(f"Loading model: {model_path}")
model = YOLO(model_path)

cap = cv2.VideoCapture(CAMERA_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Failed to open camera")
    exit(1)

for _ in range(5):
    cap.grab()

print("=" * 50)
print("CUSTOM BALL DETECTOR")
print("=" * 50)
print(f"Model: {model_path}")
print("Press 'q' to quit")
print("=" * 50)

while True:
    cap.grab()
    ret, frame = cap.retrieve()
    if not ret:
        break

    # Detect
    results = model(frame, conf=0.3, verbose=False)

    result = frame.copy()
    ball_found = False

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_name = model.names[int(box.cls[0])]

            ball_found = True

            # Draw
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Center point
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(result, (cx, cy), 6, (0, 255, 0), -1)

            # Label
            label = f"{cls_name} {conf:.0%}"
            cv2.putText(result, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Status
    status = "BALL DETECTED!" if ball_found else "Looking..."
    color = (0, 255, 0) if ball_found else (0, 0, 255)

    cv2.putText(result, status, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow('Custom Ball Detector', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
