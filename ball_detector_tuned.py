"""
Tuned Sports Ball Detector.
Uses YOLO's built-in sports ball class with optimized settings.
"""
import cv2
from ultralytics import YOLO

CAMERA_URL = "rtsp://georgedelimpasis:v!nmWiQs3Bs@192.168.1.163:554/stream2"

print("Loading YOLOv8...")
model = YOLO("yolov8n.pt")

# Sports ball = class 32
BALL_CLASS = 32

cap = cv2.VideoCapture(CAMERA_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Failed to open camera")
    exit(1)

print("\n" + "=" * 50)
print("TUNED BALL DETECTOR")
print("=" * 50)
print("Press '+' / '-' to adjust confidence")
print("Press 'q' to quit")
print("=" * 50)

conf_threshold = 0.15  # Start low

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame, conf=conf_threshold, verbose=False)

    result = frame.copy()
    ball_found = False

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if cls_id == BALL_CLASS:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                ball_found = True

                # Draw box
                cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 3)

                # Label
                label = f"BALL {conf:.0%}"
                cv2.putText(result, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Status
    status = "BALL DETECTED!" if ball_found else "Looking..."
    color = (0, 255, 0) if ball_found else (0, 0, 255)

    cv2.putText(result, status, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(result, f"Confidence: {conf_threshold:.0%} (+/-)", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('Ball Detector', result)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('+') or key == ord('='):
        conf_threshold = min(0.9, conf_threshold + 0.05)
        print(f"Confidence: {conf_threshold:.0%}")
    elif key == ord('-'):
        conf_threshold = max(0.05, conf_threshold - 0.05)
        print(f"Confidence: {conf_threshold:.0%}")

cap.release()
cv2.destroyAllWindows()

print(f"\nBest confidence threshold: {conf_threshold:.0%}")
