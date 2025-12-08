# Save as lane_stream_server.py
import time
import io
import threading
import cv2
import numpy as np
from picamera2 import Picamera2
from flask import Flask, Response

# --- Flask ---
app = Flask(__name__)
output = io.BytesIO()
lock = threading.Lock()

# --- Camera Setup ---
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))


prev_left = None
prev_right = None
alpha = 0.2   # smoothing factor (0.1?0.3 works best)


# -------------------- Lane Detection Helpers --------------------
def region_of_interest(img):
    h, w = img.shape
    polygons = np.array([[
        (0, h),
        (w, h),
        (w, int(h*0.35)),
        (0, int(h*0.35))
    ]])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    return cv2.bitwise_and(img, mask)

def make_coordinates(image, line_params):
    slope, intercept = line_params
    y1 = image.shape[0]
    y2 = int(y1 * 0.45)

    if slope == 0:
        slope = 0.0001

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    global prev_left, prev_right, alpha

    if lines is None:
        return None

    left, right = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue

        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        if abs(slope) < 0.2:
            continue

        if slope < 0:
            left.append((slope, intercept))
        else:
            right.append((slope, intercept))

    lane_lines = []

    # --- LEFT LANE ---
    if left:
        left_avg = np.mean(left, axis=0)

        if prev_left is None:
            prev_left = left_avg
        else:
            prev_left = prev_left * (1 - alpha) + left_avg * alpha

        lane_lines.append(make_coordinates(image, prev_left))

    # --- RIGHT LANE ---
    if right:
        right_avg = np.mean(right, axis=0)

        if prev_right is None:
            prev_right = right_avg
        else:
            prev_right = prev_right * (1 - alpha) + right_avg * alpha

        lane_lines.append(make_coordinates(image, prev_right))

    return lane_lines



# --- Camera Thread ---
def camera_loop():
    picam2.start_preview()
    time.sleep(1)
    picam2.start(show_preview=False)

    try:
        while True:
            # Capture to array (RGB)
            frame = picam2.capture_array()
            lane_frame = frame.copy()

            # Convert to BGR for OpenCV ops
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            
            gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=20)
            blur = cv2.GaussianBlur(gray, (5,5), 0)
            edges = cv2.Canny(blur, 30, 100)
            
            kernel = np.ones((3,3), np.uint8)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)


            masked = region_of_interest(edges)

            # Hough line detection
            lines = cv2.HoughLinesP(masked, 2, np.pi/180, 60,
                                    np.array([]), minLineLength=40, maxLineGap=5)
            averaged = average_slope_intercept(frame_bgr, lines)

            # Draw lanes
            if averaged is not None:
                for line in averaged:
                    x1, y1, x2, y2 = line
                    cv2.line(lane_frame, (x1,y1), (x2,y2), (0,255,0), 4)

            # Convert back to JPEG
            lane_bgr = cv2.cvtColor(lane_frame, cv2.COLOR_RGB2BGR)
            ret, jpeg = cv2.imencode('.jpg', lane_bgr)
            if not ret:
                continue

            with lock:
                output.seek(0)
                output.truncate()
                output.write(jpeg.tobytes())

            time.sleep(0.03)  # ~30 FPS

    finally:
        picam2.stop()

# Start background camera thread
threading.Thread(target=camera_loop, daemon=True).start()

# --- MJPEG Stream ---
def generate():
    while True:
        with lock:
            frame = output.getvalue()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.01)

@app.route('/video_feed')
def video_feed():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return ('<html><body>'
            '<h1>Lane Detection Stream</h1>'
            '<img src="/video_feed" width="640" height="480">'
            '</body></html>')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True)
