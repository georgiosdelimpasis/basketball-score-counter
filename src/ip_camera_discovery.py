"""
IP Camera Discovery Module
Scans local network to find IP Webcam devices automatically.
"""
import socket
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Dict
from dataclasses import dataclass
from config.settings import (
    IP_WEBCAM_PORT,
    IP_WEBCAM_ENDPOINTS,
    NETWORK_SCAN_TIMEOUT,
    MAX_SCAN_WORKERS
)


@dataclass
class CameraDevice:
    """Represents a discovered IP camera device"""
    ip: str
    port: int
    name: str
    device_info: Dict

    def get_url(self, mode: str = 'mjpeg', resolution: str = None) -> str:
        """
        Build camera URL for streaming.

        Args:
            mode: 'mjpeg' or 'snapshot'
            resolution: Optional resolution parameter (e.g., '480p')

        Returns:
            Full URL for camera stream
        """
        # Get endpoint from device_info (supports both iOS and Android)
        if mode == 'mjpeg':
            endpoint = self.device_info.get('mjpeg_endpoint', IP_WEBCAM_ENDPOINTS.get('mjpeg', '/video'))
        else:
            endpoint = self.device_info.get('snapshot_endpoint', IP_WEBCAM_ENDPOINTS.get('snapshot', '/shot.jpg'))

        url = f"http://{self.ip}:{self.port}{endpoint}"

        # Add resolution parameter if provided (mainly for Android IP Webcam)
        if resolution and self.device_info.get('type') == 'android':
            url += f"?{resolution}"

        return url

    def __str__(self):
        return f"{self.name} ({self.ip})"


class IPCameraDiscovery:
    """Discover IP Webcam devices on local network"""

    def __init__(self, port: int = IP_WEBCAM_PORT, timeout: int = NETWORK_SCAN_TIMEOUT):
        """
        Initialize discovery service.

        Args:
            port: Port to scan for IP Webcam (default 8080)
            timeout: Timeout for HTTP requests in seconds
        """
        self.port = port
        self.timeout = timeout

    def get_local_network_range(self) -> str:
        """
        Get local network CIDR range.

        Returns:
            Network range as string (e.g., '192.168.1')
        """
        try:
            # Create a socket to get local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # Connect to external address (doesn't actually send data)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()

            # Extract network range (assume /24 subnet)
            network_base = '.'.join(local_ip.split('.')[:-1])
            return network_base

        except Exception as e:
            print(f"Error getting local network range: {e}")
            # Fallback to common home network
            return "192.168.1"

    def probe_ip(self, ip: str) -> Optional[CameraDevice]:
        """
        Probe single IP address for IP camera apps (iPhone, Android, etc).

        Args:
            ip: IP address to probe

        Returns:
            CameraDevice if found, None otherwise
        """
        # Test 1: iPhone IP Camera Lite - check for /live endpoint
        try:
            url = f"http://{ip}:{self.port}/live"
            response = requests.get(url, timeout=self.timeout, stream=True)

            if response.status_code == 200:
                # Check if it's an MJPEG stream
                content_type = response.headers.get('content-type', '')
                if 'multipart' in content_type or 'mjpeg' in content_type:
                    return CameraDevice(
                        ip=ip,
                        port=self.port,
                        name=f"📱 iPhone Camera ({ip})",
                        device_info={
                            'type': 'ios',
                            'app': 'IP Camera Lite',
                            'mjpeg_endpoint': '/live',
                            'snapshot_endpoint': '/photo'
                        }
                    )
        except (requests.RequestException, ValueError):
            pass

        # Test 2: Android IP Webcam - check for /status.json
        try:
            status_endpoint = IP_WEBCAM_ENDPOINTS['status']
            url = f"http://{ip}:{self.port}{status_endpoint}"
            response = requests.get(url, timeout=self.timeout)

            if response.status_code == 200:
                data = response.json()

                # Verify it's IP Webcam by checking for specific fields
                if 'video_chunk_len' in data or 'curvals' in data:
                    device_name = data.get('device_name', f"📱 Android Camera ({ip})")

                    return CameraDevice(
                        ip=ip,
                        port=self.port,
                        name=device_name,
                        device_info={
                            'type': 'android',
                            'app': 'IP Webcam',
                            'mjpeg_endpoint': '/video',
                            'snapshot_endpoint': '/shot.jpg',
                            **data
                        }
                    )

        except (requests.RequestException, ValueError):
            pass

        return None

    def scan_network(self, progress_callback=None) -> List[CameraDevice]:
        """
        Scan local network for IP Webcam devices.

        Args:
            progress_callback: Optional callback function to report progress

        Returns:
            List of discovered CameraDevice objects
        """
        network_base = self.get_local_network_range()

        # Generate list of IPs to scan (x.x.x.1 to x.x.x.254)
        ips_to_scan = [f"{network_base}.{i}" for i in range(1, 255)]

        discovered_cameras = []

        # Use ThreadPoolExecutor for concurrent scanning
        with ThreadPoolExecutor(max_workers=MAX_SCAN_WORKERS) as executor:
            # Submit all probe tasks
            future_to_ip = {
                executor.submit(self.probe_ip, ip): ip
                for ip in ips_to_scan
            }

            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_ip):
                completed += 1

                # Report progress if callback provided
                if progress_callback:
                    progress_callback(completed, len(ips_to_scan))

                result = future.result()
                if result is not None:
                    discovered_cameras.append(result)

        # Sort by IP address for consistent ordering
        discovered_cameras.sort(key=lambda c: tuple(map(int, c.ip.split('.'))))

        return discovered_cameras

    def quick_scan(self, known_ips: List[str]) -> List[CameraDevice]:
        """
        Quick scan of known IP addresses (faster than full network scan).

        Args:
            known_ips: List of IP addresses to check

        Returns:
            List of discovered CameraDevice objects
        """
        discovered_cameras = []

        for ip in known_ips:
            camera = self.probe_ip(ip)
            if camera:
                discovered_cameras.append(camera)

        return discovered_cameras
