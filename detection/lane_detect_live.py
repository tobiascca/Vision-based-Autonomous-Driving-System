import pygame
from adafruit_motorkit import MotorKit
import cv2
import numpy as np

###############################################
# CAMERA SETUP (AUTO: Pi Camera or USB Camera)
###############################################
class UniversalCamera:
    def __init__(self, width=320, height=240):
        self.width = width
        self.height = height

        try:
            from picamera2 import Picamera2
            self.picam2 = Picamera2()

            config = self.picam2.create_preview_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"}
            )
            self.picam2.configure(config)
            self.picam2.start()

            self.use_picam = True
            print("[Camera] Using Picamera2")

        except Exception:
            self.use_picam = False
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

            if not self.cap.isOpened():
                raise RuntimeError("No camera found")

            print("[Camera] Using system webcam")

    def read(self):
        if self.use_picam:
            return self.picam2.capture_array()
        else:
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError("Failed to capture frame")
            return frame

    def release(self):
        if not self.use_picam:
            self.cap.release()
        cv2.destroyAllWindows()
        
###############################################
# MOTOR + KEYBOARD MOVEMENT (YOUR CODE)
###############################################
kit = MotorKit()

LOW = 0.6
TURN = 0.4
MAX = 1.0

pygame.init()
screen = pygame.display.set_mode((200, 200))
pygame.display.set_caption("RC Car Keyboard + Lane Detection")

running = True
keys = {"w": False, "s": False, "a": False, "d": False}

def clamp_motor(value):
    if value > 0:
        return max(LOW, min(MAX, value))
    elif value < 0:
        return min(-LOW, max(-MAX, value))
    else:
        return 0

def stop_motors():
    kit.motor1.throttle = 0
    kit.motor2.throttle = 0

def update_motors(lane_ok):
    motor1 = 0
    motor2 = 0

    # SAFETY: if no lane ? no forward movement
    if not lane_ok:
        stop_motors()
        return

    # Your movement logic (unchanged)
    if keys["w"]:
        motor1 = LOW + 0.04
        motor2 = LOW
    elif keys["s"]:
        motor1 = -LOW
        motor2 = -LOW

    if keys["a"]:
        motor1 -= TURN
        motor2 += TURN
    if keys["d"]:
        motor1 += TURN
        motor2 -= TURN

    kit.motor1.throttle = clamp_motor(motor1)
    kit.motor2.throttle = clamp_motor(motor2)
    
###############################################
# SIMPLE LANE PRESENCE DETECTION
###############################################
def region_of_interest(img):
    h, w = img.shape
    mask = np.zeros_like(img)

    polygon = np.array([
        [(0, h), (w, h), (w, int(h*0.55)), (0, int(h*0.55))]
    ])
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(img, mask)

def detect_lane_presence(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 150)
    cropped = region_of_interest(edges)

    lines = cv2.HoughLinesP(
        cropped,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=30,
        maxLineGap=150
    )

    # Lane is present if at least 2 segments found
    lane_ok = lines is not None and len(lines) >= 2

    # Visualize
    display = frame.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(display, (x1, y1), (x2, y2), (0,255,0), 2)

    cv2.imshow("Lane View", display)

    return lane_ok

###############################################
# MAIN LOOP
###############################################
cam = UniversalCamera()

while running:
    # ---- CAMERA ----
    frame = cam.read()
    lane_ok = detect_lane_presence(frame)

    # ---- KEYBOARD ----
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                keys["w"] = True
            elif event.key == pygame.K_s:
                keys["s"] = True
            elif event.key == pygame.K_a:
                keys["a"] = True
            elif event.key == pygame.K_d:
                keys["d"] = True
            elif event.key == pygame.K_q:
                running = False

        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_w:
                keys["w"] = False
            elif event.key == pygame.K_s:
                keys["s"] = False
            elif event.key == pygame.K_a:
                keys["a"] = False
            elif event.key == pygame.K_d:
                keys["d"] = False

    # ---- MOTOR LOGIC ----
    update_motors(lane_ok)

    # ESC closes camera window
    if cv2.waitKey(1) & 0xFF == 27:
        running = False

    pygame.time.delay(20)

# Cleanup
stop_motors()
cam.release()
pygame.quit()
