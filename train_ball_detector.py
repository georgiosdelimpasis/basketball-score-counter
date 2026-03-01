"""
Quick ball detector training.
1. Capture images by pressing SPACE
2. Click on the ball center, then drag to set size
3. Press 't' to train when you have enough images
"""
import cv2
import os
import numpy as np
from pathlib import Path

CAMERA_URL = "rtsp://georgedelimpasis:v!nmWiQs3Bs@192.168.1.163:554/stream2"

# Setup directories
dataset_dir = Path("ball_dataset")
images_dir = dataset_dir / "images" / "train"
labels_dir = dataset_dir / "labels" / "train"
images_dir.mkdir(parents=True, exist_ok=True)
labels_dir.mkdir(parents=True, exist_ok=True)

# Global state for drawing
drawing = False
start_point = None
end_point = None
current_frame = None
captured_count = 0

def mouse_callback(event, x, y, flags, param):
    global drawing, start_point, end_point

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        end_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)

def save_annotation(frame, x1, y1, x2, y2, img_id):
    """Save image and YOLO format annotation."""
    h, w = frame.shape[:2]

    # Save image
    img_path = images_dir / f"ball_{img_id:04d}.jpg"
    cv2.imwrite(str(img_path), frame)

    # Convert to YOLO format (normalized center x, center y, width, height)
    cx = ((x1 + x2) / 2) / w
    cy = ((y1 + y2) / 2) / h
    bw = abs(x2 - x1) / w
    bh = abs(y2 - y1) / h

    # Save label (class 0 = ball)
    label_path = labels_dir / f"ball_{img_id:04d}.txt"
    with open(label_path, 'w') as f:
        f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    print(f"Saved: {img_path.name} with annotation")
    return True

def create_yaml():
    """Create dataset YAML file."""
    yaml_content = f"""
path: {dataset_dir.absolute()}
train: images/train
val: images/train

names:
  0: ball
"""
    yaml_path = dataset_dir / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    return yaml_path

def train_model():
    """Train YOLO model on collected data."""
    from ultralytics import YOLO

    yaml_path = create_yaml()

    print("\n" + "=" * 60)
    print("TRAINING CUSTOM BALL DETECTOR")
    print("=" * 60)

    # Use YOLOv8 nano for fast training
    model = YOLO("yolov8n.pt")

    # Train for a few epochs
    model.train(
        data=str(yaml_path),
        epochs=50,
        imgsz=640,
        batch=8,
        name="ball_detector",
        patience=10,
        verbose=True
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    print("Model saved to: runs/detect/ball_detector/weights/best.pt")
    print("=" * 60)

# Main
cap = cv2.VideoCapture(CAMERA_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Failed to open camera")
    exit(1)

cv2.namedWindow('Train Ball Detector')
cv2.setMouseCallback('Train Ball Detector', mouse_callback)

print("=" * 60)
print("CUSTOM BALL DETECTOR TRAINER")
print("=" * 60)
print("\nInstructions:")
print("1. Press SPACE to freeze frame")
print("2. Draw a box around the ball (click and drag)")
print("3. Press 's' to save the annotation")
print("4. Press 'c' to cancel and try again")
print("5. Repeat until you have 20+ images")
print("6. Press 't' to train the model")
print("7. Press 'q' to quit")
print("=" * 60)

frozen = False
frame = None

while True:
    if not frozen:
        ret, frame = cap.read()
        if not ret:
            break
        current_frame = frame.copy()

    display = current_frame.copy()

    # Draw current box if drawing
    if start_point and end_point:
        cv2.rectangle(display, start_point, end_point, (0, 255, 0), 2)

    # Status
    status = "FROZEN - Draw box around ball" if frozen else "Press SPACE to freeze"
    color = (0, 255, 255) if frozen else (255, 255, 255)
    cv2.putText(display, status, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(display, f"Annotations: {captured_count} (need 20+)", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if captured_count >= 20:
        cv2.putText(display, "Press 't' to TRAIN!", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Train Ball Detector', display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):  # SPACE - freeze/unfreeze
        frozen = not frozen
        if frozen:
            current_frame = frame.copy()
        start_point = None
        end_point = None

    elif key == ord('s') and frozen and start_point and end_point:  # Save
        x1, y1 = start_point
        x2, y2 = end_point
        if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:  # Valid box
            save_annotation(current_frame, min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2), captured_count)
            captured_count += 1
            frozen = False
            start_point = None
            end_point = None

    elif key == ord('c'):  # Cancel
        start_point = None
        end_point = None

    elif key == ord('t') and captured_count >= 5:  # Train
        cap.release()
        cv2.destroyAllWindows()
        train_model()
        break

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if captured_count > 0 and captured_count < 20:
    print(f"\nYou captured {captured_count} images. Run again to add more, then press 't' to train.")
