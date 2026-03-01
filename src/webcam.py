"""
Webcam capture and real-time processing.
Handles camera initialization, frame capture, and cleanup.
"""
import cv2
import numpy as np
from typing import Optional, Tuple, Union
from config.settings import WEBCAM_RESOLUTIONS, DEFAULT_RESOLUTION


class WebcamCapture:
    """Manages webcam capture and frame processing."""

    def __init__(self, camera_source: Union[int, str] = 0, resolution: str = DEFAULT_RESOLUTION):
        """
        Initialize webcam capture.

        Args:
            camera_source: Camera device ID (int) for local camera (0 for default)
                          or RTSP URL (str) for IP camera (e.g., 'rtsp://user:pass@ip:554/stream1')
            resolution: Resolution preset ('320p', '640p', '1280p')
        """
        self.camera_source = camera_source
        self.resolution = resolution
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_active = False
        self.is_rtsp = isinstance(camera_source, str)

    def start(self) -> bool:
        """
        Start webcam capture.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Open camera source (local camera ID or RTSP URL)
            self.cap = cv2.VideoCapture(self.camera_source)

            if not self.cap.isOpened():
                print(f"Failed to open camera source: {self.camera_source}")
                return False

            # For local cameras, set resolution
            # RTSP streams use their native resolution
            if not self.is_rtsp:
                width, height = WEBCAM_RESOLUTIONS[self.resolution]
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                # Set FPS (may not be supported on all cameras)
                self.cap.set(cv2.CAP_PROP_FPS, 30)

            # For RTSP streams, set buffer size to reduce latency
            if self.is_rtsp:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            self.is_active = True
            return True

        except Exception as e:
            print(f"Error starting camera: {e}")
            return False

    def read(self) -> Optional[np.ndarray]:
        """
        Read a frame from the webcam.

        Returns:
            Frame as numpy array (BGR format) or None if failed
        """
        if not self.is_active or self.cap is None:
            return None

        try:
            ret, frame = self.cap.read()

            if not ret:
                return None

            return frame

        except Exception as e:
            print(f"Error reading frame: {e}")
            return None

    def stop(self):
        """Stop webcam capture and release resources."""
        self.is_active = False

        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def get_resolution(self) -> Tuple[int, int]:
        """
        Get current camera resolution.

        Returns:
            Tuple of (width, height)
        """
        if self.cap is not None and self.cap.isOpened():
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (width, height)

        return WEBCAM_RESOLUTIONS[self.resolution]

    def is_opened(self) -> bool:
        """
        Check if webcam is opened.

        Returns:
            True if opened, False otherwise
        """
        return self.cap is not None and self.cap.isOpened()

    def __del__(self):
        """Cleanup when object is destroyed."""
        self.stop()
