"""Test YOLO detection - show EVERYTHING it detects."""
import cv2
from ultralytics import YOLO

CAMERA_URL = "rtsp://georgedelimpasis:v!nmWiQs3Bs@192.168.1.163:554/stream2"

# Load YOLO model
print("Loading YOLO model...")
model = YOLO("yolov8n.pt")
print("Model loaded!")

cap = cv2.VideoCapture(CAMERA_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Failed to open camera")
    exit(1)

print("\nYOLO Detection Test")
print("Press 'q' to quit")
print("Looking for: sports ball, frisbee, or any round object...\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO with VERY low confidence to see everything
    results = model(frame, conf=0.1, verbose=False)

    result = frame.copy()

    # Draw all detections
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]

            # Color code: green for ball-like objects, gray for others
            if cls_name in ['sports ball', 'frisbee', 'orange', 'apple']:
                color = (0, 255, 0)
                thickness = 3
            else:
                color = (128, 128, 128)
                thickness = 1

            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
            label = f"{cls_name} {conf:.2f}"
            cv2.putText(result, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Print ball detections
            if cls_name in ['sports ball', 'frisbee', 'orange', 'apple']:
                print(f"🏀 DETECTED: {cls_name} (conf: {conf:.2f})")

    # Count detections
    num_detections = len(results[0].boxes) if results else 0
    cv2.putText(result, f"YOLO Detections: {num_detections} (conf > 0.1)",
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('YOLO All Detections', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
