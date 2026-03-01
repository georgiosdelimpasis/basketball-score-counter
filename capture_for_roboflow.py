"""
Capture images for Roboflow training.
Press SPACE to capture, aim for 50-100 images.
Move ball to different positions between captures!
"""
import cv2
import os
from datetime import datetime

CAMERA_URL = "rtsp://georgedelimpasis:v!nmWiQs3Bs@192.168.1.163:554/stream2"

# Create output folder
output_dir = "roboflow_images"
os.makedirs(output_dir, exist_ok=True)

# Count existing
existing = len([f for f in os.listdir(output_dir) if f.endswith('.jpg')])
count = existing

cap = cv2.VideoCapture(CAMERA_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Failed to open camera")
    exit(1)

for _ in range(5):
    cap.grab()

print("=" * 60)
print("CAPTURE IMAGES FOR ROBOFLOW")
print("=" * 60)
print(f"Output folder: {output_dir}/")
print(f"Existing images: {existing}")
print("")
print("INSTRUCTIONS:")
print("1. SPACE = Capture image")
print("2. Move ball to different positions!")
print("3. Capture 50-100 images")
print("4. Upload folder to Roboflow")
print("")
print("TIPS for good training:")
print("- Ball in corners, center, edges")
print("- Close up and far away")
print("- Different angles")
print("- With/without your hand")
print("")
print("Press 'q' to quit")
print("=" * 60)

while True:
    cap.grab()
    ret, frame = cap.retrieve()
    if not ret:
        break

    result = frame.copy()

    # Status
    cv2.rectangle(result, (10, 10), (400, 100), (0, 0, 0), -1)

    color = (0, 255, 0) if count >= 50 else (255, 255, 255)
    cv2.putText(result, f"Images: {count} / 50+", (20, 45),
               cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    if count >= 50:
        cv2.putText(result, "Ready for Roboflow!", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(result, "SPACE = Capture", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Capture for Roboflow', result)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord(' '):
        # Capture
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{output_dir}/ball_{count:04d}_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        count += 1
        print(f"Captured: {filename}")

cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 60)
print(f"DONE! Captured {count} images")
print(f"Images saved to: {output_dir}/")
print("=" * 60)
print("\nNEXT STEPS:")
print("1. Go to https://roboflow.com")
print("2. Create new project (Object Detection)")
print("3. Upload all images from 'roboflow_images' folder")
print("4. Annotate: Draw box around ball in each image")
print("5. Generate dataset (auto-augment ON)")
print("6. Train model (one click)")
print("7. Export as YOLOv8 → download best.pt")
print("=" * 60)
