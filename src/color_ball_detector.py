"""
Color-based ball detector for non-standard colored balls.
Uses HSV color filtering and contour detection.
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple


class ColorBallDetector:
    """Detects balls based on color (HSV) and circular shape."""

    def __init__(self, color_name: str = "purple"):
        """
        Initialize color ball detector.

        Args:
            color_name: Color to detect ('purple', 'orange', 'red', 'blue', 'green')
        """
        self.color_name = color_name
        self.color_ranges = {
            # Wider purple range to catch more variations (blue-purple to red-purple)
            'purple': ([110, 30, 30], [160, 255, 255]),
            'orange': ([5, 100, 100], [25, 255, 255]),
            'red_lower': ([0, 100, 100], [10, 255, 255]),
            'red_upper': ([170, 100, 100], [180, 255, 255]),
            'blue': ([100, 50, 50], [130, 255, 255]),
            'green': ([40, 50, 50], [80, 255, 255]),
            'yellow': ([20, 100, 100], [40, 255, 255]),
        }

    def detect(self, frame: np.ndarray, min_radius: int = 10, max_radius: int = 200) -> List[Dict]:
        """
        Detect colored balls in frame.

        Args:
            frame: Input frame (BGR)
            min_radius: Minimum ball radius in pixels
            max_radius: Maximum ball radius in pixels

        Returns:
            List of detections with 'box', 'class_name', 'confidence'
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Get color range
        if self.color_name in self.color_ranges:
            lower, upper = self.color_ranges[self.color_name]
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        elif self.color_name == 'red':
            # Red wraps around in HSV, need two ranges
            lower1, upper1 = self.color_ranges['red_lower']
            lower2, upper2 = self.color_ranges['red_upper']
            mask1 = cv2.inRange(hsv, np.array(lower1), np.array(upper1))
            mask2 = cv2.inRange(hsv, np.array(lower2), np.array(upper2))
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            # Default to purple
            lower, upper = self.color_ranges['purple']
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

        # Morphological operations to clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            # Filter by area
            area = cv2.contourArea(contour)
            if area < (min_radius * min_radius * 3.14):
                continue

            # Get bounding circle
            (x, y), radius = cv2.minEnclosingCircle(contour)

            if min_radius <= radius <= max_radius:
                # Calculate circularity (how round is the contour)
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * area / (perimeter * perimeter)

                # Accept less circular objects (> 0.3 for very lenient detection)
                if circularity > 0.3:
                    # Create bounding box from circle
                    x1 = int(x - radius)
                    y1 = int(y - radius)
                    x2 = int(x + radius)
                    y2 = int(y + radius)

                    # Confidence based on circularity and size
                    confidence = min(circularity, 1.0)

                    detections.append({
                        'box': [x1, y1, x2, y2],
                        'class_name': f'{self.color_name} ball',
                        'confidence': confidence,
                        'center': (int(x), int(y)),
                        'radius': int(radius)
                    })

        # Sort by area (largest first)
        detections.sort(key=lambda d: (d['box'][2] - d['box'][0]) * (d['box'][3] - d['box'][1]), reverse=True)

        return detections
