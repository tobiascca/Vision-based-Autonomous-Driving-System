from picamera2 import Picamera2
import cv2
import numpy as np
from adafruit_motorkit import MotorKit
import time

# ======================
# CAMERA CONFIG
# ======================
FRAME_W, FRAME_H = 320, 240

# ======================
# MOTOR CONFIG
# ======================
kit = MotorKit()


LOW = 0.5       # base forward speed
TURN = 0.5      # max turning strength
MAX = 1.0

KP = 0.005
KD = 0.002

last_error = 0
last_time = time.time()

# ======================
# MOTOR HELPERS
# ======================
def clamp_motor(value):
    if value > 0:
        return max(-MAX, min(MAX, value))
    #elif value < 0:
        #return min(-LOW, max(-MAX, value))
    else:
        return 0


def drive_from_lane(direction_offset):
    global last_error, last_time

    now = time.time()
    dt = max(now - last_time, 0.001)

    error = -direction_offset  # correct sign

    P = KP * error
    D = KD * (error - last_error) / dt

    steer = P + D
    steer = np.clip(steer, -0.3, 0.3)

    MIN_SPEED = 0.5  # minimum forward throttle that actually moves your motors
    base_speed = max(MIN_SPEED, LOW - abs(steer) * 0.3)

    left  = base_speed - steer
    right = base_speed + steer

    kit.motor1.throttle = clamp_motor(left)
    kit.motor2.throttle = clamp_motor(right)

    last_error = error
    last_time = now

    return steer


# ======================
# LANE DETECTION
# ======================
def region_of_interest(img):
    h, w = img.shape[:2]
    polygon = np.array([[
        (0, h),
        (0, int(h * 0.65)),
        (w, int(h * 0.65)),
        (w, h)
    ]],dtype=np.int32)
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(img, mask)


def detect_lanes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 150)
    cropped = region_of_interest(edges)

    lines = cv2.HoughLinesP(
        cropped,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=40,
        maxLineGap=100
    )

    overlay = frame.copy()
    left_x, right_x = [], []

    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            if x2 == x1:
                continue

            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.5:
                continue

            if slope < 0:
                left_x.append((x1 + x2) // 2)
                cv2.line(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
            else:
                right_x.append((x1 + x2) // 2)
                cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if left_x and right_x:
        lane_mid_x = (int(np.mean(left_x)) + int(np.mean(right_x))) // 2
        lane_detected = True
    elif left_x:
        lane_mid_x = int(np.mean(left_x)) + FRAME_W // 4
        lane_detected = True
    elif right_x:
        lane_mid_x = int(np.mean(right_x)) - FRAME_W // 4
        lane_detected = True
    else:
        lane_mid_x = FRAME_W // 2
        lane_detected = False

    direction_offset = lane_mid_x - (FRAME_W // 2)

    # Visual center line
    cv2.line(
        overlay,
        (FRAME_W // 2, FRAME_H),
        (FRAME_W // 2, FRAME_H - 50),
        (0, 0, 255),
        2
    )

    # Lane center point
    cv2.circle(
        overlay,
        (lane_mid_x, FRAME_H - 30),
        6,
        (0, 0, 255),
        -1
    )

    return edges, overlay, direction_offset, lane_detected


# ======================
# MAIN LOOP
# ======================
def main():
    picam = Picamera2()
    config = picam.create_preview_configuration(
        main={"size": (FRAME_W, FRAME_H), "format": "RGB888"}
    )
    picam.configure(config)
    picam.start()

    time.sleep(1)

    try:
        while True:
            frame = picam.capture_array()

            edges, overlay, direction_offset, lane_detected = detect_lanes(frame)
            
            if lane_detected:
                steer = drive_from_lane(direction_offset)
            else:
                steer = 0
                kit.motor1.throttle = LOW * 0.8
                kit.motor2.throttle = LOW * 0.8
                
            blue_arrow = FRAME_W // 2 + int(steer * (FRAME_W // 2))
            cv2.arrowedLine(
                overlay,
                (FRAME_W // 2, FRAME_H - 10),
                (blue_arrow, FRAME_H - 60),
                (255, 0, 0),
                3,
                tipLength=0.4
            )
            
            cv2.putText(
                overlay,
                f"offset={direction_offset} steer={steer:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            # Drive based on vision
            # drive_from_lane(direction_offset)

            # Debug windows
            cv2.imshow("Edges", edges)
            cv2.imshow("Lane Overlay", overlay)

            if cv2.waitKey(1) == 27:  # ESC
                break

    finally:
        # SAFETY STOP
        kit.motor1.throttle = 0
        kit.motor2.throttle = 0
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
