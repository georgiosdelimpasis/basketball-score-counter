"""
Test YOLO-World - detects objects by text description.
Can find "basketball" regardless of color!
"""
import cv2
from ultralytics import YOLOWorld

CAMERA_URL = "rtsp://georgedelimpasis:v!nmWiQs3Bs@192.168.1.163:554/stream2"

print("Loading YOLO-World model...")
model = YOLOWorld("yolov8s-world.pt")

# Set what we want to detect
model.set_classes(["basketball", "ball", "sphere"])
print("Model loaded! Looking for: basketball, ball, sphere")

cap = cv2.VideoCapture(CAMERA_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Failed to open camera")
    exit(1)

print("\nYOLO-World Ball Detection")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame, conf=0.1, verbose=False)

    result = frame.copy()

    # Draw detections
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]

            # Draw green box
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 3)
            label = f"{cls_name} {conf:.2f}"
            cv2.putText(result, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            print(f"Detected: {cls_name} (conf: {conf:.2f})")

    num_det = len(results[0].boxes) if results else 0
    cv2.putText(result, f"YOLO-World | Detections: {num_det}",
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('YOLO-World Ball Detection', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
