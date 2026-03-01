"""
Hybrid Ball Detector - Combines color tracking + YOLO AI detection.
Fast color tracking with AI fallback.
"""
import cv2
import numpy as np
from ultralytics import YOLO


class HybridBallDetector:
    """Hybrid ball detector: Color-based + AI detection."""

    def __init__(self):
        self.target_color = None
        self.model = None
        self.model_loaded = False

    def load_ai(self):
        """Load YOLO model for AI detection."""
        if not self.model_loaded:
            print("Loading YOLO model...")
            self.model = YOLO("yolov8n.pt")
            self.model_loaded = True
            print("YOLO ready!")

    def set_color(self, hsv_color):
        """Set target color for tracking (H, S, V)."""
        self.target_color = np.array(hsv_color)

    def sample_color(self, frame, x, y, radius=5):
        """Sample color from a point in the frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        y1, y2 = max(0, y - radius), min(hsv.shape[0], y + radius)
        x1, x2 = max(0, x - radius), min(hsv.shape[1], x + radius)
        region = hsv[y1:y2, x1:x2]
        self.target_color = np.mean(region, axis=(0, 1)).astype(int)
        return self.target_color

    def detect_by_color(self, frame):
        """Detect ball using color tracking."""
        if self.target_color is None:
            return None

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        h, s, v = self.target_color
        lower = np.array([max(0, h - 15), max(0, s - 60), max(0, v - 60)])
        upper = np.array([min(180, h + 15), 255, 255])

        mask = cv2.inRange(hsv, lower, upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_ball = None
        best_score = 0

        for c in contours:
            area = cv2.contourArea(c)
            if area < 800:
                continue

            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.5:
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(c)
            if radius < 15 or radius > 200:
                continue

            score = circularity * area
            if score > best_score:
                best_score = score
                best_ball = {
                    'center': (int(cx), int(cy)),
                    'radius': int(radius),
                    'box': [int(cx - radius), int(cy - radius),
                            int(cx + radius), int(cy + radius)],
                    'confidence': min(circularity, 1.0),
                    'method': 'color'
                }

        return best_ball

    def detect_by_ai(self, frame, conf=0.05):
        """Detect ball using YOLO AI."""
        if not self.model_loaded:
            self.load_ai()

        results = self.model(frame, conf=conf, verbose=False)

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                # Class 32 = sports ball
                if cls_id == 32:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    radius = max(x2 - x1, y2 - y1) // 2

                    return {
                        'center': (cx, cy),
                        'radius': radius,
                        'box': [x1, y1, x2, y2],
                        'confidence': float(box.conf[0]),
                        'method': 'ai'
                    }

        return None

    def detect(self, frame, prefer='color', ai_conf=0.05):
        """
        Detect ball using preferred method, fallback to other.

        Args:
            frame: BGR image
            prefer: 'color', 'ai', or 'both'
            ai_conf: AI confidence threshold

        Returns:
            Detection dict or None
        """
        if prefer == 'color':
            result = self.detect_by_color(frame)
            if result is None:
                result = self.detect_by_ai(frame, ai_conf)
        elif prefer == 'ai':
            result = self.detect_by_ai(frame, ai_conf)
            if result is None:
                result = self.detect_by_color(frame)
        else:  # both - return best
            color_det = self.detect_by_color(frame)
            ai_det = self.detect_by_ai(frame, ai_conf)

            if color_det and ai_det:
                result = color_det  # Prefer color (faster)
            else:
                result = color_det or ai_det

        return result

    @property
    def color_set(self):
        """Check if color tracking is set up."""
        return self.target_color is not None
