"""
Train Custom Ball Detector
1. Capture 10+ images of your ball
2. Auto-annotate using click
3. Train YOLO model
"""
import cv2
import os
import numpy as np
from pathlib import Path

CAMERA_URL = "rtsp://georgedelimpasis:v!nmWiQs3Bs@192.168.1.163:554/stream2"

# Setup
dataset_dir = Path("my_ball_dataset")
images_dir = dataset_dir / "images" / "train"
labels_dir = dataset_dir / "labels" / "train"
images_dir.mkdir(parents=True, exist_ok=True)
labels_dir.mkdir(parents=True, exist_ok=True)

# Count existing images
existing = len(list(images_dir.glob("*.jpg")))
img_count = existing

# State
click_point = None
ball_radius = 50  # Default radius, adjust with scroll


def mouse_callback(event, x, y, flags, param):
    global click_point
    if event == cv2.EVENT_LBUTTONDOWN:
        click_point = (x, y)


def save_annotation(frame, cx, cy, radius, img_id):
    """Save image and YOLO annotation."""
    h, w = frame.shape[:2]

    # Save image
    img_path = images_dir / f"ball_{img_id:04d}.jpg"
    cv2.imwrite(str(img_path), frame)

    # YOLO format: class cx cy w h (normalized)
    nx = cx / w
    ny = cy / h
    nw = (radius * 2) / w
    nh = (radius * 2) / h

    label_path = labels_dir / f"ball_{img_id:04d}.txt"
    with open(label_path, 'w') as f:
        f.write(f"0 {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}\n")

    return True


def create_yaml():
    """Create dataset config."""
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
    """Train YOLO on collected data."""
    from ultralytics import YOLO

    yaml_path = create_yaml()

    print("\n" + "=" * 60)
    print("TRAINING YOUR BALL DETECTOR")
    print("=" * 60)
    print(f"Training on {img_count} images...")

    model = YOLO("yolov8n.pt")

    model.train(
        data=str(yaml_path),
        epochs=100,  # More epochs for better accuracy
        imgsz=640,
        batch=8,
        name="my_ball",
        patience=20,
        verbose=True
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("Model saved to: runs/detect/my_ball/weights/best.pt")
    print("=" * 60)


# Main
cap = cv2.VideoCapture(CAMERA_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Failed to open camera")
    exit(1)

cv2.namedWindow('Train Ball')
cv2.setMouseCallback('Train Ball', mouse_callback)

for _ in range(5):
    cap.grab()

print("=" * 60)
print("CUSTOM BALL TRAINER")
print("=" * 60)
print(f"Existing images: {existing}")
print("")
print("1. Click on ball CENTER to capture")
print("2. Use +/- keys to adjust ball size")
print("3. Capture 10+ images (different positions!)")
print("4. Press 't' to train when ready")
print("")
print("CONTROLS:")
print("  Click = Capture at click position")
print("  +/=   = Increase ball size")
print("  -     = Decrease ball size")
print("  SPACE = Capture at center")
print("  't'   = Start training")
print("  'q'   = Quit")
print("=" * 60)

while True:
    cap.grab()
    ret, frame = cap.retrieve()
    if not ret:
        break

    result = frame.copy()

    # Draw guide circle at center
    h, w = frame.shape[:2]

    # If clicked, save and show feedback
    if click_point:
        cx, cy = click_point

        # Draw where we're saving
        cv2.circle(result, (cx, cy), ball_radius, (0, 255, 0), 3)
        cv2.circle(result, (cx, cy), 5, (0, 255, 0), -1)

        # Save
        save_annotation(frame, cx, cy, ball_radius, img_count)
        img_count += 1
        print(f"Saved image {img_count}")

        click_point = None

    # Show current radius guide
    cv2.putText(result, f"Ball radius: {ball_radius} (+/- to adjust)",
               (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Draw preview circle at center to show current size
    cv2.circle(result, (w // 2, h // 2), ball_radius, (0, 255, 255), 1)

    # Status
    cv2.rectangle(result, (10, 10), (350, 90), (0, 0, 0), -1)
    cv2.putText(result, f"Images: {img_count} / 10+", (20, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if img_count >= 50 else (255, 255, 255), 2)

    if img_count >= 50:
        cv2.putText(result, "Ready! Press 't' to TRAIN", (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(result, "Click on ball to capture", (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Train Ball', result)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('+') or key == ord('='):
        ball_radius = min(200, ball_radius + 5)
        print(f"Ball radius: {ball_radius}")
    elif key == ord('-') or key == ord('_'):
        ball_radius = max(20, ball_radius - 5)
        print(f"Ball radius: {ball_radius}")
    elif key == ord(' '):
        # Quick capture at center
        save_annotation(frame, w // 2, h // 2, ball_radius, img_count)
        img_count += 1
        print(f"Saved image {img_count} (center)")
    elif key == ord('t') and img_count >= 10:
        cap.release()
        cv2.destroyAllWindows()
        train_model()
        break

cap.release()
cv2.destroyAllWindows()

if img_count > 0 and img_count < 50:
    print(f"\nYou have {img_count} images. Need 10+ for good results.")
    print("Run again to add more, then press 't' to train.")
