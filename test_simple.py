"""Test simple ball detector."""
import cv2
import numpy as np
from src.simple_ball_detector import SimpleBallDetector

CAMERA_URL = "rtsp://georgedelimpasis:v!nmWiQs3Bs@192.168.1.163:554/stream2"

detector = SimpleBallDetector()
cap = cv2.VideoCapture(CAMERA_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Failed to open camera")
    exit(1)

print("Simple Ball Detector - Press 'q' to quit, 'm' to show mask")
show_mask = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect
    detections = detector.detect(frame, min_radius=30, max_radius=300)

    result = frame.copy()

    # Draw detection
    if detections:
        det = detections[0]
        cx, cy = det['center']
        radius = det['radius']
        conf = det['confidence']

        cv2.circle(result, (cx, cy), radius, (0, 255, 0), 4)
        cv2.circle(result, (cx, cy), 5, (0, 255, 0), -1)

        label = f"BALL R:{radius} C:{conf:.2f}"
        cv2.putText(result, label, (cx - radius, cy - radius - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        print(f"Detected: R={radius}, Conf={conf:.2f}, Circ={det['circularity']:.2f}")
    else:
        print("No ball detected")

    # Info
    cv2.putText(result, f"Simple Detector | Balls: {len(detections)} | Press 'm' for mask",
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('Simple Ball Detection', result)

    # Also show color mask if requested
    if show_mask:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for lower, upper in detector.color_ranges:
            m = cv2.inRange(hsv, np.array(lower), np.array(upper))
            mask = cv2.bitwise_or(mask, m)
        cv2.imshow('Color Mask', mask)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('m'):
        show_mask = not show_mask
        if not show_mask:
            cv2.destroyWindow('Color Mask')

cap.release()
cv2.destroyAllWindows()
