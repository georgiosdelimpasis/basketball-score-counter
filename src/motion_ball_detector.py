"""
Motion-based ball detector.
Detects moving round objects - works regardless of ball color.
"""
import cv2
import numpy as np
from typing import List, Dict, Optional
from collections import deque


class MotionBallDetector:
    """Detects balls by tracking motion + circular shape."""

    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100,
            varThreshold=50,
            detectShadows=False
        )
        self.prev_detections = deque(maxlen=10)

    def detect(self, frame: np.ndarray, min_radius: int = 20, max_radius: int = 300) -> List[Dict]:
        """
        Detect moving ball-like objects.

        Args:
            frame: Input frame (BGR)
            min_radius: Minimum ball radius
            max_radius: Maximum ball radius

        Returns:
            List of ball detections
        """
        h, w = frame.shape[:2]

        # Apply background subtraction to find moving objects
        fg_mask = self.bg_subtractor.apply(frame)

        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

        # Find contours in the motion mask
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []

        for contour in contours:
            area = cv2.contourArea(contour)

            # Skip tiny contours
            if area < 300:
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

            # For moving objects, we're more lenient with circularity
            # because motion blur can distort the shape
            if circularity < 0.3:
                continue

            # Calculate bounding box
            x1 = max(0, cx - radius)
            y1 = max(0, cy - radius)
            x2 = min(w - 1, cx + radius)
            y2 = min(h - 1, cy + radius)

            # Score based on circularity and size
            size_score = min(radius / 80, 1.0)  # Prefer medium-sized balls
            circ_score = circularity

            confidence = (size_score + circ_score) / 2

            candidates.append({
                'box': [x1, y1, x2, y2],
                'class_name': 'ball',
                'confidence': confidence,
                'center': (cx, cy),
                'radius': radius,
                'circularity': circularity,
                'area': area
            })

        # Sort by confidence
        candidates.sort(key=lambda x: x['confidence'], reverse=True)

        # Return best candidate if good enough
        if candidates and candidates[0]['confidence'] > 0.4:
            best = candidates[0]
            self.prev_detections.append(best)
            return [best]

        # If no motion detected, check if we had recent detections
        # (ball might have stopped briefly)
        if self.prev_detections:
            # Return last known position with lower confidence
            last = self.prev_detections[-1].copy()
            last['confidence'] *= 0.5
            return [last] if last['confidence'] > 0.3 else []

        return []

    def get_motion_mask(self, frame: np.ndarray) -> np.ndarray:
        """Get the current motion mask for debugging."""
        fg_mask = self.bg_subtractor.apply(frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        return fg_mask
