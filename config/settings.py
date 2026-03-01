"""
Configuration settings for YOLO Detection App.
Contains model definitions, parameters, and application constants.
"""
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"

# Ensure models directory exists
MODELS_DIR.mkdir(exist_ok=True)

# Custom trained model path
CUSTOM_BALL_MODEL = BASE_DIR / "runs" / "detect" / "my_ball" / "weights" / "best.pt"

# Available YOLO models with metadata
AVAILABLE_MODELS = {
    "Custom Ball (Trained)": {
        "file": str(CUSTOM_BALL_MODEL),
        "size": "18 MB",
        "speed": "Fastest",
        "map": "Custom",
        "description": "Your trained basketball detector",
        "is_custom": True
    },
    "YOLOv8n (Nano)": {
        "file": "yolov8n.pt",
        "size": "6 MB",
        "speed": "Fastest",
        "map": "37.3",
        "description": "Best for real-time webcam (10+ FPS)"
    },
    "YOLOv8s (Small)": {
        "file": "yolov8s.pt",
        "size": "22 MB",
        "speed": "Fast",
        "map": "44.9",
        "description": "Good balance of speed and accuracy"
    },
    "YOLOv8m (Medium)": {
        "file": "yolov8m.pt",
        "size": "50 MB",
        "speed": "Moderate",
        "map": "50.2",
        "description": "Higher accuracy, slower processing"
    },
    "YOLOv8l (Large)": {
        "file": "yolov8l.pt",
        "size": "84 MB",
        "speed": "Slow",
        "map": "52.9",
        "description": "Production quality detection"
    },
    "YOLOv8x (XLarge)": {
        "file": "yolov8x.pt",
        "size": "131 MB",
        "speed": "Slowest",
        "map": "53.9",
        "description": "Maximum accuracy, lowest FPS"
    }
}

# Default detection parameters
DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45
DEFAULT_IMAGE_SIZE = 640

# Webcam settings
WEBCAM_RESOLUTIONS = {
    "320p": (320, 240),
    "640p": (640, 480),
    "1280p": (1280, 720)
}
DEFAULT_RESOLUTION = "640p"
TARGET_FPS = 30

# UI constants
MAX_FILE_SIZE_MB = 200
SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "bmp", "webp"]

# Color palette for bounding boxes
COLOR_PALETTE = [
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (255, 255, 0),    # Yellow
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Cyan
    (255, 128, 0),    # Orange
    (128, 0, 255),    # Purple
    (0, 255, 128),    # Spring Green
    (255, 128, 128),  # Light Red
]

# Performance thresholds
MIN_ACCEPTABLE_FPS = 8
FPS_WARNING_THRESHOLD = 10
