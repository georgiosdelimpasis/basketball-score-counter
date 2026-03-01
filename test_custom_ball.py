"""
Test your custom-trained ball detector!
This uses the model you just trained.
"""
import cv2
from ultralytics import YOLO

CAMERA_URL = "rtsp://georgedelimpasis:v!nmWiQs3Bs@192.168.1.163:554/stream2"

# Load YOUR custom trained model
print("Loading your custom ball detector...")
model = YOLO("runs/detect/ball_detector/weights/best.pt")
print("Model loaded!")

cap = cv2.VideoCapture(CAMERA_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Failed to open camera")
    exit(1)

print("\n" + "=" * 60)
print("CUSTOM BALL DETECTOR - Using YOUR trained model!")
print("=" * 60)
print("Press 'q' to quit")
print("=" * 60)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection with your custom model
    results = model(frame, conf=0.3, verbose=False)

    result = frame.copy()

    # Draw detections
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]

            # Draw thick green box
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 4)

            # Draw label with background
            label = f"{cls_name} {conf:.0%}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(result, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), (0, 255, 0), -1)
            cv2.putText(result, label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

            print(f"BALL DETECTED! Confidence: {conf:.0%}")

    num_det = len(results[0].boxes) if results else 0
    status = "BALL DETECTED!" if num_det > 0 else "Looking for ball..."
    color = (0, 255, 0) if num_det > 0 else (255, 255, 255)

    cv2.putText(result, f"Custom Ball Detector | {status}",
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(result, "Using YOUR trained model!",
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow('Custom Ball Detection', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
