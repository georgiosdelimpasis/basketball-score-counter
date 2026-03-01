"""
Capture images and upload directly to Roboflow.
1. Edit roboflow_config.py with your API key
2. Run this script
"""
import cv2
import os
from datetime import datetime

# pip install roboflow
try:
    from roboflow import Roboflow
except ImportError:
    print("Installing roboflow package...")
    os.system("pip install roboflow")
    from roboflow import Roboflow

from roboflow_config import API_KEY, WORKSPACE, PROJECT

CAMERA_URL = "rtsp://georgedelimpasis:v!nmWiQs3Bs@192.168.1.163:554/stream2"

if API_KEY == "YOUR_API_KEY_HERE":
    print("=" * 60)
    print("Please edit roboflow_config.py with your API key!")
    print("Get your key at: https://app.roboflow.com/settings/api")
    print("=" * 60)
    exit(1)

# Connect to Roboflow
print("\nConnecting to Roboflow...")
rf = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE).project(PROJECT)
print(f"Connected to project: {PROJECT}")

# Camera
cap = cv2.VideoCapture(CAMERA_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Failed to open camera")
    exit(1)

for _ in range(5):
    cap.grab()

print("\n" + "=" * 60)
print("CAPTURE & UPLOAD TO ROBOFLOW")
print("=" * 60)
print("SPACE = Capture & Upload")
print("Move ball to different positions!")
print("'q' = Quit")
print("=" * 60)

count = 0

while True:
    cap.grab()
    ret, frame = cap.retrieve()
    if not ret:
        break

    result = frame.copy()

    # Status
    cv2.rectangle(result, (10, 10), (350, 70), (0, 0, 0), -1)
    cv2.putText(result, f"Uploaded: {count} | SPACE = Capture", (20, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Capture & Upload', result)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord(' '):
        # Save temp file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        temp_path = f"/tmp/ball_{timestamp}.jpg"
        cv2.imwrite(temp_path, frame)

        # Upload to Roboflow
        try:
            project.upload(temp_path, split="train")
            count += 1
            print(f"Uploaded image {count}")
        except Exception as e:
            print(f"Upload error: {e}")

        # Clean up
        os.remove(temp_path)

cap.release()
cv2.destroyAllWindows()

print(f"\n{'=' * 60}")
print(f"Done! Uploaded {count} images to Roboflow")
print(f"{'=' * 60}")
print("\nNext steps:")
print("1. Go to Roboflow and annotate your images")
print("2. Generate dataset")
print("3. Train model")
print("4. Export as YOLOv8")
