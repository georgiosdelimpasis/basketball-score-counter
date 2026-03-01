"""Test circle-based ball detection - should work regardless of ball color."""
import cv2
import numpy as np
from src.circle_ball_detector import HybridBallDetector

# Camera URL
CAMERA_URL = "rtsp://georgedelimpasis:v!nmWiQs3Bs@192.168.1.163:554/stream2"

# Create detector
detector = HybridBallDetector(target_color='maroon')

# Open camera
cap = cv2.VideoCapture(CAMERA_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Failed to open camera")
    exit(1)

print("Circle Ball Detector - Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect balls
    detections = detector.detect(frame, min_radius=20, max_radius=400)

    # Draw detections
    result = frame.copy()

    # Also show all detected circles for debugging
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=30, param1=100, param2=40,
        minRadius=20, maxRadius=400
    )

    # Draw all detected circles in gray
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for c in circles[0, :]:
            cv2.circle(result, (c[0], c[1]), c[2], (128, 128, 128), 2)
            cv2.circle(result, (c[0], c[1]), 3, (128, 128, 128), -1)

    # Draw best detection in green
    if detections:
        det = detections[0]
        x1, y1, x2, y2 = [int(v) for v in det['box']]
        cx, cy = det['center']
        radius = det['radius']
        conf = det['confidence']

        # Draw thick green circle
        cv2.circle(result, (cx, cy), radius, (0, 255, 0), 4)
        cv2.circle(result, (cx, cy), 5, (0, 255, 0), -1)  # Center dot

        # Label
        label = f"BALL R:{radius} C:{conf:.2f}"
        cv2.putText(result, label, (x1, max(30, y1 - 15)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        print(f"Ball detected: radius={radius}, confidence={conf:.2f}, scores={det.get('scores', {})}")
    else:
        print("No ball detected")

    # Info overlay
    num_circles = len(circles[0]) if circles is not None else 0
    cv2.putText(result, f"Circle Detector | Circles: {num_circles} | Balls: {len(detections)}",
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Circle Ball Detection', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
