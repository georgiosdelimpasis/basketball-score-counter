"""
Tune ball color with sliders - find exact HSV values.
"""
import cv2
import numpy as np

CAMERA_URL = "rtsp://georgedelimpasis:v!nmWiQs3Bs@192.168.1.163:554/stream2"

cap = cv2.VideoCapture(CAMERA_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cv2.namedWindow('Tuner')
cv2.namedWindow('Mask')

# Start with purple range
cv2.createTrackbar('H Low', 'Tuner', 120, 180, lambda x: None)
cv2.createTrackbar('H High', 'Tuner', 175, 180, lambda x: None)
cv2.createTrackbar('S Low', 'Tuner', 25, 255, lambda x: None)
cv2.createTrackbar('S High', 'Tuner', 255, 255, lambda x: None)
cv2.createTrackbar('V Low', 'Tuner', 25, 255, lambda x: None)
cv2.createTrackbar('V High', 'Tuner', 255, 255, lambda x: None)

print("BALL COLOR TUNER")
print("Adjust sliders until ONLY the ball is white in mask")
print("Press 'p' to print current values")
print("Press 'q' to quit")

for _ in range(5):
    cap.grab()

while True:
    cap.grab()
    ret, frame = cap.retrieve()
    if not ret:
        break

    h_low = cv2.getTrackbarPos('H Low', 'Tuner')
    h_high = cv2.getTrackbarPos('H High', 'Tuner')
    s_low = cv2.getTrackbarPos('S Low', 'Tuner')
    s_high = cv2.getTrackbarPos('S High', 'Tuner')
    v_low = cv2.getTrackbarPos('V Low', 'Tuner')
    v_high = cv2.getTrackbarPos('V High', 'Tuner')

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([h_low, s_low, v_low]),
                       np.array([h_high, s_high, v_high]))

    # Find contours and draw
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if area > 500:
            (cx, cy), radius = cv2.minEnclosingCircle(c)
            cv2.circle(frame, (int(cx), int(cy)), int(radius), (0, 255, 0), 2)

    cv2.imshow('Tuner', frame)
    cv2.imshow('Mask', mask)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        print(f"\nCurrent values:")
        print(f"lower = np.array([{h_low}, {s_low}, {v_low}])")
        print(f"upper = np.array([{h_high}, {s_high}, {v_high}])")

cap.release()
cv2.destroyAllWindows()

print(f"\nFinal values:")
print(f"lower = np.array([{h_low}, {s_low}, {v_low}])")
print(f"upper = np.array([{h_high}, {s_high}, {v_high}])")
