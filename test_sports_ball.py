"""
Test standard YOLO with 'sports ball' class.
Standard YOLOv8 can detect 'sports ball' (class 32).
"""
import cv2
from ultralytics import YOLO

CAMERA_URL = "rtsp://georgedelimpasis:v!nmWiQs3Bs@192.168.1.163:554/stream2"

print("Loading YOLOv8 (looking for 'sports ball' class)...")
model = YOLO("yolov8n.pt")

# Class 32 = sports ball in COCO dataset
SPORTS_BALL_CLASS = 32

cap = cv2.VideoCapture(CAMERA_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Failed to open camera")
    exit(1)

print("\n" + "=" * 60)
print("YOLO Sports Ball Detection")
print("=" * 60)
print(f"Looking for class: {model.names[SPORTS_BALL_CLASS]}")
print("Using LOW confidence threshold (0.1)")
print("Press 'q' to quit")
print("=" * 60)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection with very low confidence
    results = model(frame, conf=0.1, verbose=False)

    result = frame.copy()

    ball_detected = False

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = model.names[cls_id]

            # Check if it's a sports ball
            if cls_id == SPORTS_BALL_CLASS:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                ball_detected = True

                # Draw green box
                cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 4)
                label = f"BALL {conf:.0%}"
                cv2.putText(result, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                print(f"Sports ball detected! Confidence: {conf:.0%}")

    status = "BALL DETECTED!" if ball_detected else "Looking for sports ball..."
    color = (0, 255, 0) if ball_detected else (0, 0, 255)

    cv2.putText(result, f"YOLO Sports Ball | {status}",
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(result, "Using standard YOLO 'sports ball' class",
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('YOLO Sports Ball', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
