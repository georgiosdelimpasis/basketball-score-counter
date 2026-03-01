"""
Motion-based Object Detection.
Detects ANY moving objects using background subtraction - no training required.
Perfect for detecting non-standard colored objects like purple basketballs.
"""
import cv2
import numpy as np
from typing import List, Dict


class MotionDetector:
    """
    Detects moving objects using background subtraction.
    Works on ANY moving object regardless of color, size, or appearance.
    """

    def __init__(self):
        """Initialize motion detector."""
        # Background subtractor - automatically learns background
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=120,           # Frames to learn background
            varThreshold=25,       # Sensitivity (lower = more sensitive)
            detectShadows=True     # Remove shadows from detections
        )

        # Minimum object size to detect (in pixels)
        self.min_area = 800  # Filters out noise, keeps basketballs

        # Initialize tracking
        self.frame_count = 0

    def detect(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int = 640
    ) -> List[Dict]:
        """
        Detect moving objects in the image.

        Args:
            image: Input image (BGR format from OpenCV)
            conf_threshold: Minimum confidence (used for area-based filtering)
            iou_threshold: Not used in motion detection
            image_size: Not used in motion detection

        Returns:
            List of detection dictionaries with keys:
            'box', 'class_id', 'class_name', 'confidence'
        """
        self.frame_count += 1

        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(image)

        # Remove shadows (they appear as gray in mask)
        _, fg_mask = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)

        # Morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        # Find contours of moving objects
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []

        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by minimum area
            if area < self.min_area:
                continue

            # === CIRCLE DETECTION: Only detect ball-shaped objects ===

            # 1. Check aspect ratio (width/height should be ~1.0 for circles)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0

            # Accept objects with aspect ratio between 0.6 and 1.4 (roughly circular)
            if aspect_ratio < 0.6 or aspect_ratio > 1.4:
                continue

            # 2. Check circularity (4π * area / perimeter²)
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)

            # Accept objects with circularity > 0.5 (basketballs are ~0.7-0.9)
            # Perfect circle = 1.0, square = 0.785
            if circularity < 0.5:
                continue

            # Calculate confidence based on how circular it is
            # More circular = higher confidence
            shape_confidence = min(1.0, circularity * 1.2)  # Boost circular objects
            size_confidence = min(0.95, 0.5 + (area / 10000))
            confidence = (shape_confidence + size_confidence) / 2

            # Apply confidence threshold
            if confidence < conf_threshold:
                continue

            # Format as YOLO-style detection
            detections.append({
                'box': [x, y, x + w, y + h],  # [x1, y1, x2, y2]
                'class_id': 0,  # Single class: "ball"
                'class_name': 'ball',
                'confidence': float(confidence),
                'area': area,
                'circularity': float(circularity)  # For debugging
            })

        # Sort by confidence (largest objects first)
        detections.sort(key=lambda x: x['confidence'], reverse=True)

        return detections

    def get_model_info(self) -> Dict:
        """
        Get information about the motion detector.

        Returns:
            Dictionary with model metadata
        """
        return {
            'model_name': 'Motion Detection',
            'device': 'cpu',
            'size': '~1 KB',
            'speed': 'Very Fast',
            'description': 'Detects moving circular objects (balls only)'
        }

    def reset_background(self):
        """Reset the background model (useful if camera moves)."""
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=120,
            varThreshold=25,
            detectShadows=True
        )
        self.frame_count = 0
