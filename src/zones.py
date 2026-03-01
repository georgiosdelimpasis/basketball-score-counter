"""
Zone management for basketball scoring detection.
Allows users to draw detection zones on video and track ball movement.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import streamlit as st


class ZoneManager:
    """Manages detection zones for scoring."""

    def __init__(self, settings_file: str = "zone_settings.json"):
        self.settings_file = Path(settings_file)
        self.zones: Dict[str, Tuple[int, int, int, int]] = {}
        self.passed_zone1 = False
        self.cooldown = 0
        self.score_2pt = 0
        self.score_3pt = 0
        self.total_score = 0
        self.shot_origin_y = 0  # To track where the shot came from
        self.ball_history: List[Tuple[int, int]] = []  # Track ball positions for velocity
        self.load_zones()

    def save_zones(self):
        """Save zones to JSON file."""
        settings = {
            'zones': {name: list(coords) for name, coords in self.zones.items()}
        }
        with open(self.settings_file, 'w') as f:
            json.dump(settings, f, indent=2)

    def load_zones(self):
        """Load zones from JSON file."""
        if self.settings_file.exists():
            with open(self.settings_file, 'r') as f:
                settings = json.load(f)
                self.zones = {
                    name: tuple(coords)
                    for name, coords in settings.get('zones', {}).items()
                }

    def add_zone(self, name: str, x1: int, y1: int, x2: int, y2: int):
        """Add or update a zone."""
        self.zones[name] = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        self.save_zones()

    def remove_zone(self, name: str):
        """Remove a zone."""
        if name in self.zones:
            del self.zones[name]
            self.save_zones()

    def clear_zones(self):
        """Clear all zones."""
        self.zones.clear()
        self.save_zones()

    def point_in_zone(self, x: int, y: int, zone_name: str) -> bool:
        """Check if a point is inside a zone."""
        if zone_name not in self.zones:
            return False
        x1, y1, x2, y2 = self.zones[zone_name]
        return x1 <= x <= x2 and y1 <= y <= y2

    def check_scoring(self, detections: List[Dict]) -> bool:
        """
        Check if ball passed through Zone 1 -> Zone 2 (scoring).

        Args:
            detections: List of detection dictionaries with ball positions

        Returns:
            True if a score was detected
        """
        if 'Zone 1' not in self.zones or 'Zone 2' not in self.zones:
            return False

        # Update cooldown
        if self.cooldown > 0:
            self.cooldown -= 1

        # Find ball detection
        ball_detection = None
        for det in detections:
            if det['class_name'] == 'ball':
                # Get center point of bounding box
                x1, y1, x2, y2 = det['box']
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                ball_detection = (cx, cy)
                break

        if ball_detection is None:
            return False

        cx, cy = ball_detection

        # Track ball position history (keep last 7 positions)
        self.ball_history.append((cx, cy))
        if len(self.ball_history) > 7:
            self.ball_history.pop(0)

        in_zone1 = self.point_in_zone(cx, cy, 'Zone 1')
        in_zone2 = self.point_in_zone(cx, cy, 'Zone 2')

        # Scoring logic: Zone 1 -> Zone 2 (with velocity check)
        if in_zone1 and not self.passed_zone1 and self.cooldown == 0:
            self.passed_zone1 = True
            return False

        if in_zone2 and self.passed_zone1 and self.cooldown == 0:
            # Check ball velocity to distinguish real score from airball
            is_moving_down = self._is_ball_moving_down()

            if is_moving_down:
                # Ball is falling through net - REAL SCORE!
                self.passed_zone1 = False
                self.cooldown = 30  # 30 frames cooldown
                return True
            else:
                # Ball is moving upward or sideways - AIRBALL/RIM-OUT
                # Reset state without scoring
                self.passed_zone1 = False
                self.ball_history.clear()  # Clear history for next shot
                return False

        return False

    def _is_ball_moving_down(self) -> bool:
        """
        Check if ball is moving downward (real score) or upward (airball).

        Returns:
            True if ball is moving downward, False otherwise
        """
        if len(self.ball_history) < 3:
            # Not enough data, assume downward (allow score)
            return True

        # Calculate average vertical velocity over last few positions
        # Positive velocity = moving down (cy increasing)
        # Negative velocity = moving up (cy decreasing)

        velocities = []
        for i in range(len(self.ball_history) - 1):
            prev_cy = self.ball_history[i][1]
            curr_cy = self.ball_history[i + 1][1]
            velocity = curr_cy - prev_cy  # Positive = downward
            velocities.append(velocity)

        # Average velocity over recent frames
        avg_velocity = sum(velocities) / len(velocities)

        # If average velocity is positive (downward) or near zero, it's a score
        # Threshold: -2 pixels allows for slight upward drift at bottom of arc
        return avg_velocity > -2

    def add_manual_score(self, points: int):
        """Add a manually confirmed score."""
        if points == 3:
            self.score_3pt += 1
            self.total_score += 3
        elif points == 2:
            self.score_2pt += 1
            self.total_score += 2
        # If points == 0, it means the user cancelled the score, nothing to add

    def reset_score(self):
        """Reset the score counter."""
        self.score_2pt = 0
        self.score_3pt = 0
        self.total_score = 0
        self.passed_zone1 = False
        self.cooldown = 0
        self.ball_history.clear()  # Clear velocity tracking

    def get_zone_color(self, zone_name: str) -> Tuple[int, int, int]:
        """Get color for a zone."""
        colors = {
            'Zone 1': (255, 0, 0),    # Blue in BGR
            'Zone 2': (0, 0, 255),    # Red in BGR
            'Zone 3': (0, 255, 0),    # Green in BGR
            'Zone 4': (255, 255, 0),  # Cyan in BGR
        }
        return colors.get(zone_name, (255, 255, 255))
