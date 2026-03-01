"""
Click on the ball to get its exact color values.
"""
import cv2
import numpy as np

CAMERA_URL = "rtsp://georgedelimpasis:v!nmWiQs3Bs@192.168.1.163:554/stream1"

hsv_values = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        bgr = frame[y, x]
        h, s, v = hsv[y, x]
        hsv_values.append((h, s, v))
        print(f"\n>>> CLICK #{len(hsv_values)}")
        print(f"    HSV: H={h}, S={s}, V={v}")
        print(f"    BGR: B={bgr[0]}, G={bgr[1]}, R={bgr[2]}")

cap = cv2.VideoCapture(CAMERA_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Failed to open camera")
    exit(1)

print("=" * 50)
print("BALL COLOR CALIBRATION")
print("=" * 50)
print("\nClick on DIFFERENT parts of the ball")
print("(light areas, dark areas, edges)")
print("Click 5-10 times, then press 'q'")
print("=" * 50)

cv2.namedWindow('Calibrate')
cv2.setMouseCallback('Calibrate', mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()

    cv2.putText(display, "CLICK ON THE BALL (different spots)",
               (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(display, f"Clicks: {len(hsv_values)} (need 5+)",
               (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Resize for display
    h, w = display.shape[:2]
    if w > 1280:
        scale = 1280 / w
        display = cv2.resize(display, None, fx=scale, fy=scale)

    cv2.imshow('Calibrate', display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if hsv_values:
    print("\n" + "=" * 50)
    print("RESULTS - Use these values:")
    print("=" * 50)

    h_vals = [v[0] for v in hsv_values]
    s_vals = [v[1] for v in hsv_values]
    v_vals = [v[2] for v in hsv_values]

    h_min, h_max = min(h_vals) - 10, max(h_vals) + 10
    s_min, s_max = max(0, min(s_vals) - 30), 255
    v_min, v_max = max(0, min(v_vals) - 30), 255

    print(f"\nHue range:   {max(0,h_min)} - {min(180,h_max)}")
    print(f"Sat range:   {s_min} - {s_max}")
    print(f"Val range:   {v_min} - {v_max}")
    print(f"\nLower: np.array([{max(0,h_min)}, {s_min}, {v_min}])")
    print(f"Upper: np.array([{min(180,h_max)}, {s_max}, {v_max}])")
