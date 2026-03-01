"""Quick ball detection test - shows what's being detected and why."""
import cv2
import numpy as np
from src.color_ball_detector import ColorBallDetector

# Camera URL
CAMERA_URL = "rtsp://georgedelimpasis:v!nmWiQs3Bs@192.168.1.163:554/stream2"

# Create detector
detector = ColorBallDetector('purple')

# Open camera
cap = cv2.VideoCapture(CAMERA_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("❌ Failed to open camera")
    exit(1)

print("✅ Camera connected")
print("🟣 Looking for purple balls...")
print("Press 'q' to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect balls
    detections = detector.detect(frame, min_radius=5, max_radius=800)

    # Draw detections
    result = frame.copy()

    if detections:
        print(f"✅ BALL DETECTED! Count: {len(detections)}")
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det['box']]
            conf = det['confidence']
            radius = det['radius']

            print(f"   - Radius: {radius}px, Confidence: {conf:.2f}")

            # Draw green box for detected balls
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(result, f"BALL {conf:.2f} R:{radius}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        # Show why nothing was detected
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([110, 30, 30])
        upper = np.array([160, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        print(f"❌ No balls detected. Contours found: {len(contours)}")

        # Show all contours with details
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < 314:  # min_radius=5 → area < π*5²
                continue

            (x, y), radius = cv2.minEnclosingCircle(contour)
            perimeter = cv2.arcLength(contour, True)

            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)

                # Draw rejected contours in red
                cv2.circle(result, (int(x), int(y)), int(radius), (0, 0, 255), 2)

                status = "TOO BIG" if radius > 800 else f"C={circularity:.2f} < 0.3"
                cv2.putText(result, f"REJECTED: {status}",
                           (int(x - radius), int(y - radius - 10)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                print(f"   Contour {i}: Radius={int(radius)}px, Circularity={circularity:.2f}, Area={int(area)}")

    # Add info overlay
    cv2.putText(result, f"Purple Ball Detector | Detections: {len(detections)}",
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('Ball Detection Test', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\n👋 Test complete")
