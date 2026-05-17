from picamera2 import Picamera2
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from adafruit_motorkit import MotorKit
import time
import os

# ======================
# CAMERA CONFIG
# ======================
FRAME_W, FRAME_H = 320, 240

# ======================
# MODEL CONFIG
# ======================
MODEL_PATH = "/home/capstone/Desktop/capstone/object/tflite_learn_860574_3_v2.tflite"
LABELS = ["background", "stop", "redlight", "greenlight"]

BASE_THRESHOLD = 0.80
STOP_THRESHOLD = 0.90
BOX_SCALE = 2.0

# ======================
# MOTOR CONFIG
# ======================
kit = MotorKit()

LOW = 0.5
MAX = 1.0

KP = 0.005
KD = 0.002

last_error = 0
last_time = time.time()

STOP_TIME = 3.0
stop_until = 0

# ======================
# LOAD MODEL
# ======================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model not found")

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

in_h = input_details[0]["shape"][1]
in_w = input_details[0]["shape"][2]

# ======================
# MOTOR HELPERS
# ======================
def clamp(v):
    return max(-MAX, min(MAX, v))

def drive_from_lane(offset):
    global last_error, last_time

    now = time.time()
    dt = max(now - last_time, 0.001)

    error = -offset
    P = KP * error
    D = KD * (error - last_error) / dt
    steer = np.clip(P + D, -0.3, 0.3)

    base = max(0.45, LOW - abs(steer) * 0.3)

    kit.motor1.throttle = clamp(base - steer)
    kit.motor2.throttle = clamp(base + steer)

    last_error = error
    last_time = now
    return steer


def stop_car():
    kit.motor1.throttle = 0
    kit.motor2.throttle = 0

# ======================
# LANE DETECTION
# ======================
def region_of_interest(img):
    h, w = img.shape[:2]
    mask = np.zeros_like(img)
    poly = np.array([[
        (0, h),
        (0, int(h * 0.65)),
        (w, int(h * 0.65)),
        (w, h)
    ]], dtype=np.int32)
    cv2.fillPoly(mask, poly, 255)
    return cv2.bitwise_and(img, mask)

def detect_lanes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 150)
    cropped = region_of_interest(edges)

    lines = cv2.HoughLinesP(
        cropped, 1, np.pi/180, 50,
        minLineLength=40, maxLineGap=100
    )

    left, right = [], []

    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            if x1 == x2:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.5:
                continue
            if slope < 0:
                left.append((x1 + x2) // 2)
            else:
                right.append((x1 + x2) // 2)

    if left and right:
        mid = (int(np.mean(left)) + int(np.mean(right))) // 2
        detected = True
    else:
        mid = FRAME_W // 2
        detected = False

    offset = mid - FRAME_W // 2
    return offset, detected

# ======================
# FOMO DETECTION (IMPROVED)
# ======================
def detect_objects(frame, overlay):
    global stop_until

    img = cv2.resize(frame, (in_w, in_h)).astype(np.float32) / 255.0
    in_scale, in_zero = input_details[0]["quantization"]
    img = (img / in_scale + in_zero).astype(np.int8)
    img = np.expand_dims(img, 0)

    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])[0]
    out_scale, out_zero = output_details[0]["quantization"]
    output = (output.astype(np.float32) - out_zero) * out_scale

    gh, gw, _ = output.shape
    cw, ch = FRAME_W / gw, FRAME_H / gh

    best = None

    for y in range(gh):
        for x in range(gw):
            cid = int(np.argmax(output[y, x]))
            score = float(output[y, x, cid])

            if cid == 0:
                continue

            label = LABELS[cid]
            threshold = STOP_THRESHOLD if label == "stop" else BASE_THRESHOLD

            if score < threshold:
                continue

            cx = int((x + 0.5) * cw)
            cy = int((y + 0.5) * ch)

            bw = int(cw * BOX_SCALE)
            bh = int(ch * BOX_SCALE)

            x1 = max(0, cx - bw // 2)
            y1 = max(0, cy - bh // 2)
            x2 = min(FRAME_W, cx + bw // 2)
            y2 = min(FRAME_H, cy + bh // 2)

            area = (x2 - x1) * (y2 - y1)

            # Logic filters
            if label == "stop" and area < 1500:
                continue
            if label in ("redlight", "greenlight") and y2 > FRAME_H * 0.6:
                continue

            best = (label, score, x1, y1, x2, y2)

    if best:
        label, score, x1, y1, x2, y2 = best

        if label == "stop":
            stop_until = time.time() + STOP_TIME

        color = (0, 255, 0) if label == "greenlight" else (0, 0, 255)

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            overlay,
            f"{label} {score:.2f}",
            (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

        return label

    return None
    
# ======================
# MAIN LOOP
# ======================
picam = Picamera2()
picam.configure(
    picam.create_preview_configuration(
        main={"size": (FRAME_W, FRAME_H), "format": "RGB888"}
    )
)
picam.start()
time.sleep(1)

try:
    while True:
        frame = picam.capture_array()
        overlay = frame.copy()

        detected = detect_objects(frame, overlay)

        if time.time() < stop_until:
            stop_car()
            cv2.putText(overlay, "STOP",
                        (120, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 0, 255), 3)
        else:
            offset, lane_ok = detect_lanes(frame)
            if lane_ok:
                drive_from_lane(offset)
            else:
                kit.motor1.throttle = LOW * 0.8
                kit.motor2.throttle = LOW * 0.8

        cv2.imshow("Vision", overlay)
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    stop_car()
    cv2.destroyAllWindows()
    picam.close()
