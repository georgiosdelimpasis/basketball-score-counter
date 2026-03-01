"""
Click on the ball to get its exact HSV color values.
This will help us detect it properly.
"""
import cv2
import numpy as np

CAMERA_URL = "rtsp://georgedelimpasis:v!nmWiQs3Bs@192.168.1.163:554/stream2"

# Store clicked colors
clicked_colors = []

def mouse_callback(event, x, y, flags, param):
    """Handle mouse clicks to sample color."""
    global frame, clicked_colors

    if event == cv2.EVENT_LBUTTONDOWN:
        if frame is not None:
            # Get BGR color at click point
            bgr = frame[y, x]

            # Convert to HSV
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv = hsv_frame[y, x]

            print(f"\n🎯 CLICKED AT ({x}, {y})")
            print(f"   BGR: {bgr}")
            print(f"   HSV: H={hsv[0]}, S={hsv[1]}, V={hsv[2]}")

            clicked_colors.append(hsv)

            # Calculate suggested range based on all clicks
            if len(clicked_colors) >= 1:
                all_h = [c[0] for c in clicked_colors]
                all_s = [c[1] for c in clicked_colors]
                all_v = [c[2] for c in clicked_colors]

                h_min, h_max = max(0, min(all_h) - 10), min(180, max(all_h) + 10)
                s_min, s_max = max(0, min(all_s) - 30), min(255, max(all_s) + 30)
                v_min, v_max = max(0, min(all_v) - 30), min(255, max(all_v) + 30)

                print(f"\n📋 SUGGESTED COLOR RANGE (copy this):")
                print(f"   Lower: [{h_min}, {s_min}, {v_min}]")
                print(f"   Upper: [{h_max}, {s_max}, {v_max}]")

cap = cv2.VideoCapture(CAMERA_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Failed to open camera")
    exit(1)

cv2.namedWindow('Click on the Ball')
cv2.setMouseCallback('Click on the Ball', mouse_callback)

print("=" * 50)
print("🏀 BALL COLOR CALIBRATION")
print("=" * 50)
print("\nInstructions:")
print("1. Click directly on your basketball (multiple times in different spots)")
print("2. The HSV values will be printed")
print("3. After a few clicks, copy the SUGGESTED COLOR RANGE")
print("4. Press 'r' to reset, 'q' to quit")
print("=" * 50)

frame = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()

    # Draw crosshair at center
    h, w = frame.shape[:2]

    # Show instructions
    cv2.putText(display, "CLICK ON THE BALL to sample its color", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display, f"Samples collected: {len(clicked_colors)}", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(display, "Press 'r' to reset, 'q' to quit", (10, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow('Click on the Ball', display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        clicked_colors = []
        print("\n🔄 Reset - click on the ball again")

cap.release()
cv2.destroyAllWindows()

# Print final summary
if clicked_colors:
    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)

    all_h = [c[0] for c in clicked_colors]
    all_s = [c[1] for c in clicked_colors]
    all_v = [c[2] for c in clicked_colors]

    h_min, h_max = max(0, min(all_h) - 10), min(180, max(all_h) + 10)
    s_min, s_max = max(0, min(all_s) - 30), min(255, max(all_s) + 30)
    v_min, v_max = max(0, min(all_v) - 30), min(255, max(all_v) + 30)

    print(f"\nCopy this color range:")
    print(f"  LOWER = [{h_min}, {s_min}, {v_min}]")
    print(f"  UPPER = [{h_max}, {s_max}, {v_max}]")
    print("\nPaste these values when prompted!")
