"""
Pro Ball Detector - Larger model + YOLO built-in tracking.
"""
import cv2
from ultralytics import YOLO

CAMERA_URL = "rtsp://georgedelimpasis:v!nmWiQs3Bs@192.168.1.163:554/stream2"

print("Loading YOLOv8s (larger, more accurate)...")
model = YOLO("yolov8s.pt")

BALL_CLASS = 32

cap = cv2.VideoCapture(CAMERA_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Failed to open camera")
    exit(1)

print("\n" + "=" * 50)
print("PRO BALL DETECTOR")
print("=" * 50)
print("YOLOv8s + Built-in Tracking")
print("Press 'q' to quit")
print("=" * 50)

conf_threshold = 0.05

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = frame.copy()
    ball_detected = False

    # Run YOLO with tracking
    results = model.track(frame, conf=conf_threshold, persist=True, verbose=False)

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if cls_id == BALL_CLASS:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                ball_detected = True

                # Get track ID if available
                track_id = ""
                if box.id is not None:
                    track_id = f" ID:{int(box.id[0])}"

                # Draw detection (green)
                cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 3)
                label = f"BALL {conf:.0%}{track_id}"
                cv2.putText(result, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Draw center point
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(result, (cx, cy), 6, (0, 255, 0), -1)

    # Status
    status = "BALL DETECTED!" if ball_detected else "Looking for ball..."
    color = (0, 255, 0) if ball_detected else (0, 0, 255)

    cv2.putText(result, status, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(result, "YOLOv8s + Tracking", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('Pro Ball Detector', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
