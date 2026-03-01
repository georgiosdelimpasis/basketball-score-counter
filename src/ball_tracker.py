"""
Basketball tracking and scoring logic.
Tracks ball position and detects when it goes through the hoop.
"""
import numpy as np
from typing import Optional, Tuple, List, Dict
from collections import deque


class BasketballTracker:
    """Tracks basketball position and detects scoring."""

    def __init__(self, max_history: int = 30):
        """
        Initialize basketball tracker.

        Args:
            max_history: Number of frames to keep in history
        """
        self.max_history = max_history
        self.position_history = deque(maxlen=max_history)
        self.hoop_zone = None  # (x1, y1, x2, y2)
        self.score = 0
        self.last_state = None  # 'above', 'in', 'below', None
        self.scoring_cooldown = 0  # Prevent double counting

    def set_hoop_zone(self, x1: int, y1: int, x2: int, y2: int):
        """
        Set the hoop scoring zone.

        Args:
            x1, y1: Top-left corner
            x2, y2: Bottom-right corner
        """
        self.hoop_zone = (x1, y1, x2, y2)

    def get_ball_center(self, ball_box: List[float]) -> Tuple[int, int]:
        """Get center point of ball bounding box."""
        x1, y1, x2, y2 = ball_box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        return cx, cy

    def is_in_hoop_zone(self, cx: int, cy: int) -> bool:
        """Check if ball center is within hoop zone."""
        if self.hoop_zone is None:
            return False
        x1, y1, x2, y2 = self.hoop_zone
        return x1 <= cx <= x2 and y1 <= cy <= y2

    def get_ball_state(self, cx: int, cy: int) -> str:
        """
        Determine ball state relative to hoop.

        Returns:
            'above', 'in', or 'below'
        """
        if self.hoop_zone is None:
            return None

        x1, y1, x2, y2 = self.hoop_zone

        # Check if horizontally aligned with hoop
        if x1 <= cx <= x2:
            if cy < y1:
                return 'above'
            elif y1 <= cy <= y2:
                return 'in'
            else:
                return 'below'
        return None

    def update(self, ball_detections: List[Dict]) -> bool:
        """
        Update tracker with new ball detection.

        Args:
            ball_detections: List of ball detections (should be 1 or 0)

        Returns:
            True if a score was detected, False otherwise
        """
        # Decrease cooldown
        if self.scoring_cooldown > 0:
            self.scoring_cooldown -= 1

        # If no ball detected or hoop not set, reset state
        if len(ball_detections) == 0 or self.hoop_zone is None:
            self.last_state = None
            return False

        # Get first ball (assume only one ball)
        ball = ball_detections[0]
        cx, cy = self.get_ball_center(ball['box'])

        # Add to history
        self.position_history.append((cx, cy))

        # Get current state
        current_state = self.get_ball_state(cx, cy)

        # Check for scoring sequence: above → in → below
        scored = False
        if self.scoring_cooldown == 0:
            if (self.last_state == 'in' and current_state == 'below'):
                # Check if ball was recently above
                if len(self.position_history) >= 10:
                    # Look back to see if ball was above in last 10 frames
                    recent_states = []
                    for i in range(max(0, len(self.position_history) - 10), len(self.position_history)):
                        px, py = self.position_history[i]
                        state = self.get_ball_state(px, py)
                        if state:
                            recent_states.append(state)

                    # If we have above → in → below sequence, it's a score!
                    if 'above' in recent_states:
                        self.score += 1
                        scored = True
                        self.scoring_cooldown = 30  # 1 second cooldown at 30 FPS

        self.last_state = current_state
        return scored

    def get_trajectory_line(self, num_points: int = 10) -> List[Tuple[int, int]]:
        """Get recent trajectory points for visualization."""
        if len(self.position_history) < 2:
            return []
        return list(self.position_history)[-num_points:]

    def reset_score(self):
        """Reset the score counter."""
        self.score = 0
        self.position_history.clear()
        self.last_state = None
        self.scoring_cooldown = 0
