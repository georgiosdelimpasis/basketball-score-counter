"""
Simple ball detector - combines color filtering with shape detection.
Focused and practical approach for basketball detection.
"""
import cv2
import numpy as np
from typing import List, Dict


class SimpleBallDetector:
    """
    Simple, focused ball detector.
    Works by finding the largest reddish-brown circular object.
    """

    def __init__(self):
        """Initialize detector."""
        # Your basketball appears to be maroon/burgundy with tan stripes
        # In HSV: maroon is around H=0-15 (red-ish), S=50-200, V=50-200
        self.color_ranges = [
            # Maroon/burgundy (red-brown)
            ([0, 40, 40], [15, 255, 200]),
            # Also check higher hue range for red
            ([165, 40, 40], [180, 255, 200]),
            # Brown tones
            ([10, 40, 40], [25, 200, 200]),
        ]

    def detect(self, frame: np.ndarray, min_radius: int = 30, max_radius: int = 300) -> List[Dict]:
        """
        Detect basketball in frame.

        Returns:
            List with single best detection (or empty list)
        """
        h, w = frame.shape[:2]

        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create combined mask for all color ranges
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        for lower, upper in self.color_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Clean up mask
        kernel = np.ones((7, 7), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []

        for contour in contours:
            area = cv2.contourArea(contour)

            # Skip tiny contours
            if area < 500:
                continue

            # Get enclosing circle
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            cx, cy, radius = int(cx), int(cy), int(radius)

            # Skip if outside size range
            if radius < min_radius or radius > max_radius:
                continue

            # Calculate circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)

            # Skip if not circular enough (basketball should be round)
            if circularity < 0.5:
                continue

            # Calculate how well the contour fills the circle
            circle_area = np.pi * radius * radius
            fill_ratio = area / circle_area

            # Score based on multiple factors
            score = 0.0

            # Circularity score (0-1)
            score += circularity * 0.4

            # Size score - prefer medium-large balls
            if 50 <= radius <= 200:
                score += 0.3
            elif 30 <= radius < 50 or 200 < radius <= 300:
                score += 0.15

            # Fill ratio score
            score += min(fill_ratio, 1.0) * 0.3

            x1 = max(0, cx - radius)
            y1 = max(0, cy - radius)
            x2 = min(w - 1, cx + radius)
            y2 = min(h - 1, cy + radius)

            candidates.append({
                'box': [x1, y1, x2, y2],
                'class_name': 'ball',
                'confidence': score,
                'center': (cx, cy),
                'radius': radius,
                'circularity': circularity,
                'fill_ratio': fill_ratio
            })

        # Sort by score and return best match
        candidates.sort(key=lambda x: x['confidence'], reverse=True)

        # Only return if we have a good candidate
        if candidates and candidates[0]['confidence'] > 0.5:
            return [candidates[0]]

        return []
