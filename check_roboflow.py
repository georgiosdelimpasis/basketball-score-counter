"""Check Roboflow workspace and projects."""
from roboflow import Roboflow
from roboflow_config import API_KEY

rf = Roboflow(api_key=API_KEY)

print("=" * 60)
print("YOUR ROBOFLOW ACCOUNT INFO")
print("=" * 60)

# Get workspace info
workspace = rf.workspace()
print(f"\nWorkspace ID: {workspace.name}")
print(f"Workspace URL: {workspace.url}")

print("\n" + "-" * 40)
print("YOUR PROJECTS:")
print("-" * 40)

projects = workspace.project_list
if projects:
    for p in projects:
        print(f"  - {p['name']} (id: {p['id']})")
else:
    print("  (No projects yet - create one on roboflow.com)")

print("\n" + "=" * 60)
print("NEXT STEPS:")
print("=" * 60)
print("1. Go to: https://app.roboflow.com")
print("2. Click 'Create New Project'")
print("3. Name: basketball-detector")
print("4. Type: Object Detection")
print("5. Then run capture_upload_roboflow.py again")
print("=" * 60)
