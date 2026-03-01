"""
Basketball tracking and scoring logic.
Tracks ball position and detects when it goes through the hoop.
"""
import numpy as np
from typing import Optional, Tuple, List, Dict
from collections import deque


class BasketballTracker:
    """Tracks basketball position and detects scoring using Line Intersection geometry."""

    def __init__(self, max_history: int = 30):
        self.max_history = max_history
        self.position_history = deque(maxlen=max_history)
        self.hoop_zone = None  # (hx1, hy1, hx2, hy2)
        self.score_2pt = 0
        self.score_3pt = 0
        self.total_score = 0
        self.scoring_cooldown = 0
        self.zones = {}
        
        # Physics state
        self.potential_score_frames = 0
        self.entered_rim_x = None

    def set_hoop_zone(self, x1: int, y1: int, x2: int, y2: int):
        self.hoop_zone = (x1, y1, x2, y2)

    def set_zones(self, zones: Dict):
        self.zones = zones

    def get_ball_center(self, ball_box: List[float]) -> Tuple[int, int]:
        x1, y1, x2, y2 = ball_box
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def _do_lines_intersect(self, p1, p2, p3, p4):
        """Check if line segment p1-p2 intersects p3-p4."""
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

    def update(self, ball_detections: List[Dict]) -> Tuple[bool, int]:
        # Decrease cooldown
        if self.scoring_cooldown > 0:
            self.scoring_cooldown -= 1

        if len(ball_detections) == 0 or self.hoop_zone is None:
            # If we were tracking a potential score but the ball disappeared, it might have fallen out the bottom
            if self.potential_score_frames > 0:
                self.potential_score_frames -= 1
            return False, 0

        # Get the ball with the highest confidence
        # (Helps ignore random noise heads/shoes if there are multiple "balls")
        best_ball = max(ball_detections, key=lambda b: b.get('confidence', 0))
        cx, cy = self.get_ball_center(best_ball['box'])

        # Filter out massive jumps (indicates YOLO grabbed the wrong object randomly)
        if len(self.position_history) > 0:
            last_cx, last_cy = self.position_history[-1]
            dist = np.sqrt((cx - last_cx)**2 + (cy - last_cy)**2)
            if dist > 300: # Pixel jump threshold (too fast is physically impossible)
                return False, 0 # Ignore it completely

        self.position_history.append((cx, cy))

        scored = False
        points_awarded = 0

        if self.scoring_cooldown == 0 and len(self.position_history) >= 2:
            hx1, hy1, hx2, hy2 = self.hoop_zone
            rim_left = (hx1, hy1)
            rim_right = (hx2, hy1)
            net_bottom_left = (hx1, hy2)
            net_bottom_right = (hx2, hy2)

            p1 = self.position_history[-2] # Previous frame point
            p2 = self.position_history[-1] # Current frame point

            # 1. Check if the ball vector crosses the top rim (Downwards)
            if p1[1] <= hy1 and p2[1] >= hy1:
                if self._do_lines_intersect(p1, p2, rim_left, rim_right):
                    # Enter! The ball mathematically breached the rim from above.
                    # Wait up to 15 frames for it to fall out the bottom of the net box.
                    self.potential_score_frames = 15
                    self.entered_rim_x = cx

            # 2. Check if a ball currently "in the net" falls out the bottom.
            if self.potential_score_frames > 0:
                self.potential_score_frames -= 1
                
                # If it crosses the bottom line (y2) or just exists below it, and stays horizontally constrained
                if p2[1] >= hy2:
                    # To prove it didn't just fly sideways behind the backboard,
                    # Check if its X coordinate is still inside or very close to the net box bounds
                    margin = (hx2 - hx1) * 0.5 # 50% wiggle room
                    if (hx1 - margin) <= cx <= (hx2 + margin):
                        # SWISH! It went through the top and fell out the bottom!
                        scored = True

            # 3. Super High-Speed Fallback (Flashing through both top and bottom in 1 frame)
            if not scored and p1[1] < hy1 and p2[1] > hy2:
                if self._do_lines_intersect(p1, p2, rim_left, rim_right):
                    # It plummeted blindly fast.
                    scored = True

            if scored:
                # Find release point (where the shot began 30 frames ago)
                rx, ry = self.position_history[0]
                points_awarded = 2 # Default

                if self.zones:
                    for zone_name, (zx1, zy1, zx2, zy2) in self.zones.items():
                        if zx1 <= rx <= zx2 and zy1 <= ry <= zy2:
                            if "3" in zone_name or "Zone 2" in zone_name:
                                points_awarded = 3
                            else:
                                points_awarded = 2
                            break

                if points_awarded == 3:
                    self.score_3pt += 1
                    self.total_score += 3
                else:
                    self.score_2pt += 1
                    self.total_score += 2

                self.scoring_cooldown = 45 # 1.5 second cooldown
                self.potential_score_frames = 0 # reset state

        return scored, points_awarded

    def get_trajectory_line(self, num_points: int = 15) -> List[Tuple[int, int]]:
        if len(self.position_history) < 2:
            return []
        return list(self.position_history)[-num_points:]

    def reset_score(self):
        self.score_2pt = 0
        self.score_3pt = 0
        self.total_score = 0
        self.position_history.clear()
        self.scoring_cooldown = 0
        self.potential_score_frames = 0
