"""Test Roboflow model detection."""
import cv2
from roboflow import Roboflow
from roboflow_config import API_KEY, WORKSPACE, PROJECT

CAMERA_URL = "rtsp://georgedelimpasis:v!nmWiQs3Bs@192.168.1.163:554/stream2"

print("Connecting to Roboflow...")
rf = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE).project(PROJECT)

# List available versions
print("\nAvailable versions:")
try:
    for v in project.versions:
        print(f"  Version {v.version}: {v.id}")
except:
    pass

# Try to get latest version
print("\nTrying to load model...")
try:
    # Try version 1
    model = project.version(1).model
    print("Model loaded: version 1")
except Exception as e:
    print(f"Version 1 failed: {e}")
    try:
        # Try version 2
        model = project.version(2).model
        print("Model loaded: version 2")
    except Exception as e2:
        print(f"Version 2 failed: {e2}")
        print("\nCould not load model. Check Roboflow dashboard for available versions.")
        exit(1)

# Capture one frame and test
print("\nCapturing test frame...")
cap = cv2.VideoCapture(CAMERA_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

for _ in range(5):
    cap.grab()

ret, frame = cap.read()
cap.release()

if not ret:
    print("Failed to capture frame")
    exit(1)

# Save test frame
test_path = "/tmp/test_ball.jpg"
cv2.imwrite(test_path, frame)
print(f"Saved test frame to {test_path}")

# Run inference
print("\nRunning inference...")
try:
    result = model.predict(test_path, confidence=10, overlap=30).json()
    print(f"\nRaw result: {result}")

    predictions = result.get('predictions', [])
    print(f"\nFound {len(predictions)} detections:")

    for i, pred in enumerate(predictions):
        print(f"  {i+1}. Class: {pred.get('class', 'unknown')}")
        print(f"      Confidence: {pred.get('confidence', 0):.1%}")
        print(f"      Position: ({pred.get('x', 0)}, {pred.get('y', 0)})")
        print(f"      Size: {pred.get('width', 0)}x{pred.get('height', 0)}")

except Exception as e:
    print(f"Inference error: {e}")
    print("\nTrying alternative inference method...")

    # Try using inference URL directly
    try:
        import requests
        import base64

        with open(test_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode()

        # Get model endpoint
        print(f"Project: {PROJECT}")
        print("Check your Roboflow dashboard for the inference endpoint")
    except Exception as e2:
        print(f"Alternative method also failed: {e2}")

print("\n" + "=" * 50)
print("If no detections, possible issues:")
print("1. Model not deployed - go to Roboflow → Deploy")
print("2. Need more training images")
print("3. Confidence too high - try lower threshold")
print("=" * 50)
