"""Download trained model from Roboflow."""
from roboflow import Roboflow
from roboflow_config import API_KEY, WORKSPACE, PROJECT

rf = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE).project(PROJECT)

print("Available versions:")
for v in project.versions:
    print(f"  Version {v.version}")

# Get latest version
version = project.version(1)  # Change number if needed

print(f"\nDownloading model version {version.version}...")
model = version.model

print("\nModel ready!")
print("You can now run: python3 use_roboflow_model.py")
print("\nOr use the model directly:")
print("  from roboflow import Roboflow")
print("  rf = Roboflow(api_key=API_KEY)")
print("  model = rf.workspace().project(PROJECT).version(1).model")
print("  prediction = model.predict('image.jpg', confidence=40)")
