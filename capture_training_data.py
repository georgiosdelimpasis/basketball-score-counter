"""
Capture training images of your ball.
Press SPACE to capture, 'q' when done (need ~20-30 images).
"""
import cv2
import os
from datetime import datetime

CAMERA_URL = "rtsp://georgedelimpasis:v!nmWiQs3Bs@192.168.1.163:554/stream2"

# Create directories
os.makedirs("training_data/images", exist_ok=True)

cap = cv2.VideoCapture(CAMERA_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Failed to open camera")
    exit(1)

print("=" * 60)
print("TRAINING DATA CAPTURE")
print("=" * 60)
print("\nInstructions:")
print("1. Hold the ball in different positions")
print("2. Press SPACE to capture an image")
print("3. Move the ball around, rotate it, different distances")
print("4. Capture 20-30 images from different angles")
print("5. Press 'q' when done")
print("=" * 60)

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()

    # Show instructions
    cv2.putText(display, f"Images captured: {count}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(display, "SPACE = capture, Q = done", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(display, "Move ball to different positions!", (10, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow('Capture Training Data', display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):  # SPACE
        filename = f"training_data/images/ball_{count:04d}.jpg"
        cv2.imwrite(filename, frame)
        count += 1
        print(f"Captured: {filename}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n{'=' * 60}")
print(f"Done! Captured {count} images.")
print(f"Images saved to: training_data/images/")
print(f"\nNext step: Run the labeling tool to mark the ball in each image.")
print(f"{'=' * 60}")
