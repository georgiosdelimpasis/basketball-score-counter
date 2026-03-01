"""
Detect round objects - works for any ball color.
Uses edge detection + circularity, not color.
"""
import cv2
import numpy as np

CAMERA_URL = "rtsp://georgedelimpasis:v!nmWiQs3Bs@192.168.1.163:554/stream1"

def detect_balls(frame):
    """Find circular objects using edge detection."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Dilate edges to connect gaps
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    balls = []

    for contour in contours:
        area = cv2.contourArea(contour)

        # Filter by size (adjust for 2K resolution)
        if area < 5000 or area > 500000:
            continue

        # Get enclosing circle
        (cx, cy), radius = cv2.minEnclosingCircle(contour)

        # Skip if too small or too big
        if radius < 40 or radius > 400:
            continue

        # Calculate circularity
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)

        # Must be reasonably circular
        if circularity < 0.5:
            continue

        # Calculate how well contour fills the circle
        circle_area = np.pi * radius * radius
        fill = area / circle_area

        balls.append({
            'center': (int(cx), int(cy)),
            'radius': int(radius),
            'circularity': circularity,
            'fill': fill,
            'score': circularity * fill
        })

    # Sort by score
    balls.sort(key=lambda x: x['score'], reverse=True)

    return balls[:3], edges  # Return top 3 candidates


cap = cv2.VideoCapture(CAMERA_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Failed to open camera")
    exit(1)

print("=" * 50)
print("ROUND OBJECT DETECTOR")
print("=" * 50)
print("Detects circular shapes - any color ball")
print("Press 'e' to show edges")
print("Press 'q' to quit")
print("=" * 50)

show_edges = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    balls, edges = detect_balls(frame)

    result = frame.copy()

    # Draw all candidates
    for i, ball in enumerate(balls):
        cx, cy = ball['center']
        radius = ball['radius']
        score = ball['score']

        # Green for best, yellow for others
        color = (0, 255, 0) if i == 0 else (0, 255, 255)
        thickness = 6 if i == 0 else 3

        cv2.circle(result, (cx, cy), radius, color, thickness)
        cv2.circle(result, (cx, cy), 8, color, -1)

        label = f"#{i+1} {score:.0%}"
        cv2.putText(result, label, (cx - radius, cy - radius - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    status = f"Found {len(balls)} round object(s)" if balls else "No round objects"
    color = (0, 255, 0) if balls else (0, 0, 255)

    cv2.putText(result, status, (10, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # Resize for display
    h, w = result.shape[:2]
    if w > 1280:
        scale = 1280 / w
        result = cv2.resize(result, None, fx=scale, fy=scale)
        edges = cv2.resize(edges, None, fx=scale, fy=scale)

    cv2.imshow('Round Object Detector', result)

    if show_edges:
        cv2.imshow('Edges', edges)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('e'):
        show_edges = not show_edges
        if not show_edges:
            cv2.destroyWindow('Edges')

cap.release()
cv2.destroyAllWindows()
