"""
Fast Ball Detector - Optimized for low latency.
"""
import cv2
from ultralytics import YOLO

CAMERA_URL = "rtsp://georgedelimpasis:v!nmWiQs3Bs@192.168.1.163:554/stream2"

model = YOLO("yolov8n.pt")  # Nano = fastest
BALL_CLASS = 32

cap = cv2.VideoCapture(CAMERA_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Flush buffer to get latest frame
for _ in range(5):
    cap.grab()

print("FAST BALL DETECTOR | Press 'q' to quit")

while True:
    # Grab and flush old frames for lowest latency
    cap.grab()
    ret, frame = cap.retrieve()
    if not ret:
        break

    # Detect
    results = model(frame, conf=0.05, verbose=False)

    # Draw only ball detections
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            if int(box.cls[0]) == BALL_CLASS:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(frame, f"BALL {conf:.0%}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Ball', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
