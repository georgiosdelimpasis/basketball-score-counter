"""
Ball Tracker App - Color + AI hybrid detection.
Click on ball to enable fast color tracking.
AI detection as fallback.
"""
import cv2
from src.hybrid_detector import HybridBallDetector

CAMERA_URL = "rtsp://georgedelimpasis:v!nmWiQs3Bs@192.168.1.163:554/stream2"

detector = HybridBallDetector()
mode = 'ai'  # Start with AI, switch to color after click

def mouse_callback(event, x, y, flags, param):
    global mode
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv = detector.sample_color(frame, x, y)
        mode = 'color'
        print(f"Color set: H={hsv[0]} S={hsv[1]} V={hsv[2]}")
        print("Switched to COLOR tracking (fast)")

cap = cv2.VideoCapture(CAMERA_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cv2.namedWindow('Ball Tracker')
cv2.setMouseCallback('Ball Tracker', mouse_callback)

# Flush buffer
for _ in range(5):
    cap.grab()

print("=" * 50)
print("BALL TRACKER")
print("=" * 50)
print("Click on ball = enable fast color tracking")
print("Press 'a' = AI only mode")
print("Press 'c' = Color only mode")
print("Press 'b' = Both (color + AI fallback)")
print("Press 'r' = Reset color")
print("Press 'q' = Quit")
print("=" * 50)

# Pre-load AI model
detector.load_ai()

frame = None

while True:
    cap.grab()
    ret, frame = cap.retrieve()
    if not ret:
        break

    # Detect ball
    detection = detector.detect(frame, prefer=mode, ai_conf=0.05)

    result = frame.copy()

    if detection:
        cx, cy = detection['center']
        radius = detection['radius']
        conf = detection['confidence']
        method = detection['method']

        # Green = color, Yellow = AI
        color = (0, 255, 0) if method == 'color' else (0, 255, 255)

        cv2.circle(result, (cx, cy), radius, color, 3)
        cv2.circle(result, (cx, cy), 5, color, -1)

        label = f"BALL {conf:.0%} [{method.upper()}]"
        cv2.putText(result, label, (cx - radius, cy - radius - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        status = "DETECTED"
        status_color = color
    else:
        status = "Looking..."
        status_color = (0, 0, 255)

    # Info bar
    cv2.putText(result, f"Mode: {mode.upper()} | {status}",
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    if not detector.color_set:
        cv2.putText(result, "Click on ball for fast tracking",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('Ball Tracker', result)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('a'):
        mode = 'ai'
        print("Mode: AI only")
    elif key == ord('c'):
        mode = 'color'
        print("Mode: Color only")
    elif key == ord('b'):
        mode = 'both'
        print("Mode: Both (color + AI fallback)")
    elif key == ord('r'):
        detector.target_color = None
        mode = 'ai'
        print("Color reset - back to AI mode")

cap.release()
cv2.destroyAllWindows()
