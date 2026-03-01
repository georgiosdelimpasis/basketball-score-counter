"""
Webcam capture and real-time processing.
Handles camera initialization, frame capture, and cleanup.
"""
import cv2
import numpy as np
import threading
import time
import requests
from typing import Optional, Tuple, Union
from collections import deque
from config.settings import WEBCAM_RESOLUTIONS, DEFAULT_RESOLUTION


class WebcamCapture:
    """Manages webcam capture and frame processing."""

    def __init__(self, camera_source: Union[int, str] = 2, resolution: str = DEFAULT_RESOLUTION, streaming_mode: str = 'mjpeg'):
        """
        Initialize webcam capture.

        Args:
            camera_source: Camera device ID (int) for local camera (0 for default)
                          or RTSP URL (str) for IP camera (e.g., 'rtsp://user:pass@ip:554/stream1')
                          or HTTP URL (str) for IP Webcam (e.g., 'http://192.168.1.136:8080/video')
            resolution: Resolution preset ('320p', '640p', '1280p')
            streaming_mode: 'mjpeg' for continuous stream or 'snapshot' for polling mode
        """
        self.camera_source = camera_source
        self.resolution = resolution
        self.streaming_mode = streaming_mode
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_active = False
        self.is_rtsp = isinstance(camera_source, str) and camera_source.startswith('rtsp://')
        self.is_http = isinstance(camera_source, str) and camera_source.startswith('http://')
        self.latest_frame = None
        self.lock = threading.Lock()
        self.thread = None

        # Performance tracking
        self.frame_times = deque(maxlen=30)  # Track last 30 frame times for FPS calculation
        self.latency_ms = 0.0  # Frame capture latency in milliseconds

        # Snapshot mode specific
        self.snapshot_url = None
        self.http_session = None

    def start(self) -> bool:
        """
        Start webcam capture.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Handle snapshot mode for HTTP URLs
            if self.is_http and self.streaming_mode == 'snapshot':
                # Extract base URL and use snapshot endpoint
                if '/video' in self.camera_source:
                    self.snapshot_url = self.camera_source.replace('/video', '/shot.jpg')
                elif '/shot.jpg' in self.camera_source:
                    self.snapshot_url = self.camera_source
                else:
                    # Assume base URL, add /shot.jpg
                    self.snapshot_url = self.camera_source.rstrip('/') + '/shot.jpg'

                # Create persistent HTTP session for connection pooling
                self.http_session = requests.Session()

                # Test the connection
                try:
                    response = self.http_session.get(self.snapshot_url, timeout=5)
                    if response.status_code != 200:
                        print(f"Failed to connect to snapshot URL: {self.snapshot_url}")
                        return False
                except Exception as e:
                    print(f"Error testing snapshot connection: {e}")
                    return False

                self.is_active = True

                # Start background thread for snapshot polling
                self.thread = threading.Thread(target=self._update_snapshot, daemon=True)
                self.thread.start()

                return True

            else:
                # Standard OpenCV capture for local cameras, RTSP, or MJPEG streams
                self.cap = cv2.VideoCapture(self.camera_source)

                if not self.cap.isOpened():
                    print(f"Failed to open camera source: {self.camera_source}")
                    return False

                # For local cameras, set resolution
                # RTSP and HTTP streams use their native resolution
                if not self.is_rtsp and not self.is_http:
                    width, height = WEBCAM_RESOLUTIONS[self.resolution]
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    # Set FPS (may not be supported on all cameras)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)

                # For RTSP streams, set buffer size to reduce latency
                if self.is_rtsp:
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                # For HTTP MJPEG streams, also set buffer size
                if self.is_http:
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                self.is_active = True

                # Start background thread to continually pull frames and defeat the OpenCV buffer
                self.thread = threading.Thread(target=self._update, daemon=True)
                self.thread.start()

                return True

        except Exception as e:
            print(f"Error starting camera: {e}")
            return False

    def _update(self):
        """Continuously grab frames from the camera in the background to prevent FFMPEG buffer buildup."""
        while self.is_active and self.cap is not None:
            start_time = time.time()
            ret, frame = self.cap.read()
            read_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            if ret:
                with self.lock:
                    self.latest_frame = frame
                    self.latency_ms = read_time
                    self.frame_times.append(time.time())

    def _update_snapshot(self):
        """Continuously poll snapshot endpoint for maximum FPS (snapshot mode)."""
        while self.is_active and self.http_session is not None:
            try:
                start_time = time.time()
                response = self.http_session.get(self.snapshot_url, timeout=0.5)
                read_time = (time.time() - start_time) * 1000  # Convert to milliseconds

                if response.status_code == 200:
                    # Decode JPEG image from response
                    img_array = np.frombuffer(response.content, dtype=np.uint8)
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                    if frame is not None:
                        with self.lock:
                            self.latest_frame = frame
                            self.latency_ms = read_time
                            self.frame_times.append(time.time())

            except Exception as e:
                # Suppress errors to keep polling
                pass

            # Small delay to prevent CPU overload (allows ~100 FPS max polling rate)
            time.sleep(0.01)

    def read(self) -> Optional[np.ndarray]:
        """
        Read a frame from the webcam.

        Returns:
            Frame as numpy array (BGR format) or None if failed
        """
        if not self.is_active:
            return None

        # Wait briefly for the background thread to pull the very first frame
        for _ in range(50):
            with self.lock:
                frame = self.latest_frame
            if frame is not None:
                break
            time.sleep(0.1)

        try:
            if frame is None:
                return None

            # Removed rotation check so the app respects phone's native orientation

            # Calculate exactly proportional dimensions to keep perfect aspect ratio (e.g., 9:16)
            h, w = frame.shape[:2]
            target_max_size, _ = WEBCAM_RESOLUTIONS[self.resolution]
            
            # Scale the video down based on its longest edge so Streamlit stays fast
            scale = target_max_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
                
            frame = cv2.resize(frame, (new_w, new_h))

            return frame

        except Exception as e:
            print(f"Error reading frame: {e}")
            return None

    def get_current_fps(self) -> float:
        """
        Calculate current FPS from recent frame times.

        Returns:
            Current FPS as float, or 0.0 if not enough data
        """
        if len(self.frame_times) < 2:
            return 0.0

        time_span = self.frame_times[-1] - self.frame_times[0]
        if time_span > 0:
            return len(self.frame_times) / time_span
        return 0.0

    def stop(self):
        """Stop webcam capture and release resources."""
        self.is_active = False

        if hasattr(self, 'thread') and self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        if self.http_session is not None:
            self.http_session.close()
            self.http_session = None

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
