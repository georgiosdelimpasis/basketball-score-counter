"""
Utility functions for visualization and statistics.
Includes bounding box drawing, color generation, FPS calculation.
"""
import cv2
import numpy as np
import time
from typing import List, Dict, Tuple
from config.settings import COLOR_PALETTE


def generate_class_colors(num_classes: int = 80) -> Dict[int, Tuple[int, int, int]]:
    """
    Generate distinct colors for each class.

    Args:
        num_classes: Number of classes to generate colors for

    Returns:
        Dictionary mapping class IDs to BGR colors
    """
    colors = {}
    for i in range(num_classes):
        colors[i] = COLOR_PALETTE[i % len(COLOR_PALETTE)]
    return colors


def draw_bounding_boxes(
    frame: np.ndarray,
    detections: List[Dict],
    colors: Dict[int, Tuple[int, int, int]]
) -> np.ndarray:
    """
    Draw bounding boxes with labels on the frame.

    Args:
        frame: Input frame (BGR format)
        detections: List of detection dictionaries with keys:
                   'box', 'class_id', 'class_name', 'confidence'
        colors: Dictionary mapping class IDs to colors

    Returns:
        Annotated frame with bounding boxes
    """
    annotated = frame.copy()

    for det in detections:
        # Extract detection info
        box = det['box']
        class_id = det['class_id']
        class_name = det['class_name']
        confidence = det['confidence']

        # Get box coordinates
        x1, y1, x2, y2 = map(int, box)

        # Get color for this class
        color = colors.get(class_id, (0, 255, 0))

        # Draw rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Create label
        label = f"{class_name} {confidence:.2%}"

        # Calculate label size
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )

        # Draw label background
        cv2.rectangle(
            annotated,
            (x1, y1 - label_height - baseline - 5),
            (x1 + label_width, y1),
            color,
            -1
        )

        # Draw label text
        cv2.putText(
            annotated,
            label,
            (x1, y1 - baseline - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    return annotated


def calculate_fps(start_time: float, frame_count: int) -> float:
    """
    Calculate current FPS.

    Args:
        start_time: Time when counting started
        frame_count: Number of frames processed

    Returns:
        Current FPS
    """
    elapsed_time = time.time() - start_time
    if elapsed_time > 0:
        return frame_count / elapsed_time
    return 0.0


def format_detection_stats(detections: List[Dict]) -> Dict:
    """
    Calculate statistics from detections.

    Args:
        detections: List of detection dictionaries

    Returns:
        Dictionary with statistics
    """
    if not detections:
        return {
            'total_objects': 0,
            'class_counts': {},
            'avg_confidence': 0.0,
            'max_confidence': 0.0,
            'min_confidence': 0.0
        }

    # Count objects by class
    class_counts = {}
    confidences = []

    for det in detections:
        class_name = det['class_name']
        confidence = det['confidence']

        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        confidences.append(confidence)

    return {
        'total_objects': len(detections),
        'class_counts': class_counts,
        'avg_confidence': np.mean(confidences),
        'max_confidence': np.max(confidences),
        'min_confidence': np.min(confidences)
    }


def add_fps_overlay(frame: np.ndarray, fps: float) -> np.ndarray:
    """
    Add FPS counter overlay to frame.

    Args:
        frame: Input frame
        fps: Current FPS value

    Returns:
        Frame with FPS overlay
    """
    annotated = frame.copy()

    # FPS text
    fps_text = f"FPS: {fps:.1f}"

    # Position at top-right corner
    height, width = frame.shape[:2]
    (text_width, text_height), baseline = cv2.getTextSize(
        fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
    )

    x = width - text_width - 10
    y = text_height + 10

    # Draw background
    cv2.rectangle(
        annotated,
        (x - 5, y - text_height - 5),
        (x + text_width + 5, y + baseline + 5),
        (0, 0, 0),
        -1
    )

    # Draw text
    color = (0, 255, 0) if fps >= 10 else (0, 165, 255) if fps >= 5 else (0, 0, 255)
    cv2.putText(
        annotated,
        fps_text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA
    )

    return annotated


def draw_zones(frame: np.ndarray, zone_manager) -> np.ndarray:
    """
    Draw detection zones on the frame.

    Args:
        frame: Input frame
        zone_manager: ZoneManager instance with zones

    Returns:
        Frame with zones drawn
    """
    annotated = frame.copy()

    for zone_name, (x1, y1, x2, y2) in zone_manager.zones.items():
        # Get zone color
        color = zone_manager.get_zone_color(zone_name)

        # Draw zone rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Draw zone label
        cv2.putText(
            annotated,
            zone_name,
            (x1 + 5, y1 + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA
        )

    return annotated


def add_score_overlay(frame: np.ndarray, score: int) -> np.ndarray:
    """
    Add score counter overlay to frame.

    Args:
        frame: Input frame
        score: Current score

    Returns:
        Frame with score overlay
    """
    annotated = frame.copy()
    height, width = frame.shape[:2]

    # Score text
    score_text = f"SCORE: {score}"

    # Position at top-left corner
    (text_width, text_height), baseline = cv2.getTextSize(
        score_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3
    )

    x = 10
    y = text_height + 10

    # Draw background
    cv2.rectangle(
        annotated,
        (x - 5, y - text_height - 5),
        (x + text_width + 10, y + baseline + 10),
        (0, 0, 0),
        -1
    )

    # Draw text
    cv2.putText(
        annotated,
        score_text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3,
        cv2.LINE_AA
    )

    return annotated
