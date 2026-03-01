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

# Available YOLO models with metadata
AVAILABLE_MODELS = {
    "Motion Detection (No Training)": {
        "file": "motion",  # Special flag for motion detector
        "size": "~1 KB",
        "speed": "Very Fast",
        "map": "N/A",
        "description": "Detects moving circular objects only (any color ball!)",
        "is_motion": True
    },
    "YOLO11n (Nano)": {
        "file": "yolo11n.pt",
        "size": "5.3 MB",
        "speed": "Fastest",
        "map": "39.5",
        "description": "Newest generation (11). Extreme speed with improved accuracy."
    },
    "YOLO11s (Small)": {
        "file": "yolo11s.pt",
        "size": "18.2 MB",
        "speed": "Fast",
        "map": "47.0",
        "description": "Newest generation (11). More accurate for small/far basketballs."
    },
    "YOLO11m (Medium)": {
        "file": "yolo11m.pt",
        "size": "40.3 MB",
        "speed": "Moderate",
        "map": "51.5",
        "description": "High accuracy, but may reduce your camera's FPS."
    },
    "YOLO11l (Large)": {
        "file": "yolo11l.pt",
        "size": "50.1 MB",
        "speed": "Slow",
        "map": "53.4",
        "description": "Maximum accuracy. Best for pre-recorded videos, not real-time."
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

# IP Camera settings
IP_WEBCAM_PORT = 8080
IP_WEBCAM_ENDPOINTS = {
    'mjpeg': '/video',
    'snapshot': '/shot.jpg',
    'status': '/status.json'
}
IP_CAMERA_RESOLUTIONS = {
    "360p": "640x360",
    "480p": "854x480",
    "640p": "1024x640",
    "720p": "1280x720"
}
NETWORK_SCAN_TIMEOUT = 2  # seconds
MAX_SCAN_WORKERS = 50  # concurrent IP probes

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
