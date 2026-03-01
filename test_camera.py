#!/usr/bin/env python3
"""
Quick camera connection test script.
Tests various RTSP URL formats for Tapo C220.
"""
import cv2
import sys

def test_rtsp_connection(rtsp_url, timeout=10):
    """Test if RTSP URL connects successfully."""
    print(f"\n🔍 Testing: {rtsp_url}")
    print("-" * 60)

    try:
        cap = cv2.VideoCapture(rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            print("❌ Failed to open stream")
            return False

        print("✅ Stream opened successfully!")

        # Try to read a frame
        print("📹 Reading frame...")
        ret, frame = cap.read()

        if ret:
            print(f"✅ Frame captured! Size: {frame.shape}")
            cap.release()
            return True
        else:
            print("❌ Could not read frame from stream")
            cap.release()
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    camera_ip = "192.168.1.163"
    password = "v!nmWiQs3Bs"
    password_encoded = "v%21nmWiQs3Bs"  # ! → %21

    # Test different URL formats with both username formats
    test_urls = [
        # With short username
        f"rtsp://georgedelimpasis:{password}@{camera_ip}:554/stream1",
        f"rtsp://georgedelimpasis:{password}@{camera_ip}:554/stream2",

        # With email as username
        f"rtsp://georgedelimpasis@gmail.com:{password}@{camera_ip}:554/stream1",
        f"rtsp://georgedelimpasis@gmail.com:{password}@{camera_ip}:554/stream2",

        # URL-encoded password with short username
        f"rtsp://georgedelimpasis:{password_encoded}@{camera_ip}:554/stream1",
        f"rtsp://georgedelimpasis:{password_encoded}@{camera_ip}:554/stream2",

        # URL-encoded password with email username
        f"rtsp://georgedelimpasis@gmail.com:{password_encoded}@{camera_ip}:554/stream1",
        f"rtsp://georgedelimpasis@gmail.com:{password_encoded}@{camera_ip}:554/stream2",

        # Try admin as fallback
        f"rtsp://admin:{password}@{camera_ip}:554/stream1",
        f"rtsp://admin:{password}@{camera_ip}:554/stream2",

        # Without authentication
        f"rtsp://{camera_ip}:554/stream1",
        f"rtsp://{camera_ip}:554/stream2",

        # Alternative paths
        f"rtsp://georgedelimpasis:{password}@{camera_ip}:554/live",
        f"rtsp://georgedelimpasis:{password}@{camera_ip}:554/h264",
    ]

    print("=" * 60)
    print("🎥 Tapo C220 RTSP Connection Test")
    print("=" * 60)

    successful_urls = []

    for url in test_urls:
        if test_rtsp_connection(url):
            successful_urls.append(url)
            print("\n🎉 SUCCESS! This URL works!")
            print(f"Use this URL in your app: {url}")
            print("\n" + "=" * 60)
            break  # Stop after first success

    if not successful_urls:
        print("\n" + "=" * 60)
        print("❌ No working RTSP URL found")
        print("\n📋 Troubleshooting steps:")
        print("1. Verify RTSP is enabled in Tapo app")
        print("2. Check Camera Account settings for correct username/password")
        print("3. Ensure camera firmware is up to date")
        print("4. Try connecting with VLC Media Player first")
        print("5. Check if camera is on the same network")
        print("=" * 60)

if __name__ == "__main__":
    main()
