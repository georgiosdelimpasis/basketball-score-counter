"""Test motion-based ball detection - MOVE THE BALL to detect it!"""
import cv2
from src.motion_ball_detector import MotionBallDetector

CAMERA_URL = "rtsp://georgedelimpasis:v!nmWiQs3Bs@192.168.1.163:554/stream2"

detector = MotionBallDetector()
cap = cv2.VideoCapture(CAMERA_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Failed to open camera")
    exit(1)

print("=" * 50)
print("MOTION BALL DETECTOR")
print("=" * 50)
print("\n>>> MOVE THE BALL to detect it! <<<")
print("\nPress 'm' to show motion mask")
print("Press 'q' to quit")
print("=" * 50)

show_mask = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect moving balls
    detections = detector.detect(frame, min_radius=20, max_radius=300)

    result = frame.copy()

    # Draw detections
    if detections:
        det = detections[0]
        cx, cy = det['center']
        radius = det['radius']
        conf = det['confidence']

        # Draw green circle around detected ball
        cv2.circle(result, (cx, cy), radius, (0, 255, 0), 4)
        cv2.circle(result, (cx, cy), 5, (0, 255, 0), -1)

        label = f"BALL R:{radius} C:{conf:.2f}"
        cv2.putText(result, label, (cx - radius, cy - radius - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        print(f"Ball detected! R={radius}, Conf={conf:.2f}")

    # Info overlay
    status = "BALL DETECTED!" if detections else "Move the ball..."
    color = (0, 255, 0) if detections else (0, 0, 255)
    cv2.putText(result, f"Motion Detector | {status}",
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(result, "MOVE THE BALL to detect it!",
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('Motion Ball Detection', result)

    # Show motion mask if requested
    if show_mask:
        mask = detector.get_motion_mask(frame)
        cv2.imshow('Motion Mask (white = movement)', mask)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('m'):
        show_mask = not show_mask
        if not show_mask:
            cv2.destroyWindow('Motion Mask (white = movement)')

cap.release()
cv2.destroyAllWindows()
