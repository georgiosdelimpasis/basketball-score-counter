"""
Click on the ball to track it - super simple.
"""
import cv2
import numpy as np

CAMERA_URL = "rtsp://georgedelimpasis:v!nmWiQs3Bs@192.168.1.163:554/stream2"

target_color = None
tracking = False

def mouse_callback(event, x, y, flags, param):
    global target_color, tracking
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Sample 10x10 region
        y1, y2 = max(0, y-5), min(hsv.shape[0], y+5)
        x1, x2 = max(0, x-5), min(hsv.shape[1], x+5)
        region = hsv[y1:y2, x1:x2]
        target_color = np.mean(region, axis=(0,1)).astype(int)
        tracking = True
        print(f"Tracking color: H={target_color[0]} S={target_color[1]} V={target_color[2]}")

cap = cv2.VideoCapture(CAMERA_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cv2.namedWindow('Click to Track')
cv2.setMouseCallback('Click to Track', mouse_callback)

for _ in range(5):
    cap.grab()

print("CLICK ON THE BALL to start tracking!")
print("Press 'r' to reset, 'q' to quit")

frame = None

while True:
    cap.grab()
    ret, frame = cap.retrieve()
    if not ret:
        break

    result = frame.copy()

    if tracking and target_color is not None:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create range around target color
        h, s, v = target_color
        lower = np.array([max(0, h-15), max(0, s-60), max(0, v-60)])
        upper = np.array([min(180, h+15), 255, 255])

        mask = cv2.inRange(hsv, lower, upper)

        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find best circular contour
        best_ball = None
        best_score = 0

        for c in contours:
            area = cv2.contourArea(c)
            if area < 800:  # Min size
                continue

            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue

            # Check circularity - must be round!
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.5:  # Must be at least 50% circular
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(c)

            # Check size (ball should be reasonable size)
            if radius < 20 or radius > 200:
                continue

            score = circularity * area
            if score > best_score:
                best_score = score
                best_ball = (int(cx), int(cy), int(radius))

        if best_ball:
            cx, cy, radius = best_ball
            cv2.circle(result, (cx, cy), radius, (0, 255, 0), 3)
            cv2.circle(result, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(result, "BALL", (cx - 25, cy - radius - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.putText(result, "TRACKING", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(result, "CLICK ON THE BALL", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow('Click to Track', result)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        tracking = False
        target_color = None
        print("Reset - click on ball again")

cap.release()
cv2.destroyAllWindows()
