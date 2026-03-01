"""
Circle-based ball detector using Hough Circle Transform.
Detects balls by shape (circles) rather than color.
"""
import cv2
import numpy as np
from typing import List, Dict


class CircleBallDetector:
    """Detects balls based on circular shape using Hough Circle Transform."""

    def __init__(self):
        """Initialize circle ball detector."""
        pass

    def detect(self, frame: np.ndarray, min_radius: int = 20, max_radius: int = 300) -> List[Dict]:
        """
        Detect circular objects (balls) in frame.

        Args:
            frame: Input frame (BGR)
            min_radius: Minimum ball radius in pixels
            max_radius: Maximum ball radius in pixels

        Returns:
            List of detections with 'box', 'class_name', 'confidence'
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # Detect circles using Hough Circle Transform
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,  # Inverse ratio of accumulator resolution
            minDist=50,  # Minimum distance between circle centers
            param1=100,  # Upper threshold for edge detection
            param2=40,  # Threshold for circle detection (lower = more circles)
            minRadius=min_radius,
            maxRadius=max_radius
        )

        detections = []

        if circles is not None:
            circles = np.uint16(np.around(circles))

            for circle in circles[0, :]:
                cx, cy, radius = circle[0], circle[1], circle[2]

                # Create bounding box from circle
                x1 = int(cx - radius)
                y1 = int(cy - radius)
                x2 = int(cx + radius)
                y2 = int(cy + radius)

                # Calculate a confidence score based on how well-defined the circle is
                # Larger circles in the center of frame get higher confidence
                h, w = frame.shape[:2]
                center_dist = np.sqrt((cx - w/2)**2 + (cy - h/2)**2)
                max_dist = np.sqrt((w/2)**2 + (h/2)**2)
                center_score = 1 - (center_dist / max_dist)  # 1 if center, 0 if corner

                # Size score - prefer medium to large balls
                size_score = min(radius / 100, 1.0)  # Max at radius 100

                confidence = (center_score + size_score) / 2

                detections.append({
                    'box': [x1, y1, x2, y2],
                    'class_name': 'ball',
                    'confidence': confidence,
                    'center': (int(cx), int(cy)),
                    'radius': int(radius)
                })

        # Sort by radius (largest first) - biggest circle is most likely the ball
        detections.sort(key=lambda d: d['radius'], reverse=True)

        # Return only the top detection (most likely the ball)
        return detections[:1] if detections else []


class HybridBallDetector:
    """
    Hybrid detector combining multiple methods for robust ball detection.
    Uses circle detection + motion detection + optional color hints.
    """

    def __init__(self, target_color: str = 'maroon'):
        """
        Initialize hybrid ball detector.

        Args:
            target_color: Color hint for the ball ('maroon', 'orange', 'any')
        """
        self.target_color = target_color
        self.prev_frame = None
        self.prev_circles = []

        # Color ranges for verification (HSV)
        self.color_hints = {
            'maroon': ([0, 50, 30], [20, 255, 200]),  # Reddish-brown
            'orange': ([5, 100, 100], [25, 255, 255]),
            'any': ([0, 0, 0], [180, 255, 255]),  # Accept any color
        }

    def detect(self, frame: np.ndarray, min_radius: int = 15, max_radius: int = 400) -> List[Dict]:
        """
        Detect ball using multiple methods combined.

        Args:
            frame: Input frame (BGR)
            min_radius: Minimum ball radius in pixels
            max_radius: Maximum ball radius in pixels

        Returns:
            List of detections (usually 0 or 1 ball)
        """
        h, w = frame.shape[:2]
        detections = []

        # Method 1: Circle detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # Try multiple parameter combinations for robustness
        all_circles = []

        for param2 in [30, 40, 50]:  # Different sensitivity levels
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=max(30, min_radius),
                param1=100,
                param2=param2,
                minRadius=min_radius,
                maxRadius=max_radius
            )
            if circles is not None:
                all_circles.extend(circles[0])

        # Remove duplicate circles (same center within threshold)
        unique_circles = []
        for c in all_circles:
            c = list(c)  # Convert to list for easier handling
            is_duplicate = False
            for i, uc in enumerate(unique_circles):
                dist = np.sqrt((c[0] - uc[0])**2 + (c[1] - uc[1])**2)
                if dist < 30:  # Within 30 pixels = same circle
                    # Keep the larger one
                    if c[2] > uc[2]:
                        unique_circles[i] = c
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_circles.append(c)

        # Score each circle
        for circle in unique_circles:
            cx, cy, radius = int(circle[0]), int(circle[1]), int(circle[2])

            # Skip circles that are too close to edges
            if cx - radius < 0 or cy - radius < 0 or cx + radius >= w or cy + radius >= h:
                continue

            # Calculate scores
            scores = {}

            # Size score - prefer balls between 30-150 pixels radius
            if 30 <= radius <= 150:
                scores['size'] = 1.0
            elif 15 <= radius < 30:
                scores['size'] = 0.7
            elif 150 < radius <= 300:
                scores['size'] = 0.8
            else:
                scores['size'] = 0.5

            # Position score - balls are usually in upper/middle of frame during play
            y_ratio = cy / h
            if 0.1 <= y_ratio <= 0.7:
                scores['position'] = 1.0
            else:
                scores['position'] = 0.6

            # Color verification (optional)
            if self.target_color in self.color_hints:
                lower, upper = self.color_hints[self.target_color]

                # Extract region around circle
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(mask, (cx, cy), radius, 255, -1)

                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                color_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                overlap = cv2.bitwise_and(mask, color_mask)

                color_ratio = np.sum(overlap > 0) / max(np.sum(mask > 0), 1)
                scores['color'] = color_ratio if color_ratio > 0.1 else 0.3
            else:
                scores['color'] = 0.5

            # Combined confidence
            confidence = (scores['size'] * 0.4 + scores['position'] * 0.3 + scores['color'] * 0.3)

            x1 = max(0, cx - radius)
            y1 = max(0, cy - radius)
            x2 = min(w - 1, cx + radius)
            y2 = min(h - 1, cy + radius)

            detections.append({
                'box': [x1, y1, x2, y2],
                'class_name': 'ball',
                'confidence': confidence,
                'center': (cx, cy),
                'radius': radius,
                'scores': scores
            })

        # Sort by confidence and return best match
        detections.sort(key=lambda d: d['confidence'], reverse=True)

        # Return top detection if confidence is reasonable
        if detections and detections[0]['confidence'] > 0.4:
            return [detections[0]]

        return []
