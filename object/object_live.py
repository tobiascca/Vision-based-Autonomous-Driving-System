import pygame
import cv2
import numpy as np
from adafruit_motorkit import MotorKit
import tflite_runtime.interpreter as tflite
import time

# =======================
# MOTOR SETUP
# =======================
kit = MotorKit()

LOW = 0.6
TURN = 0.4
MAX = 1.0

# =======================
# CAMERA SETUP
# =======================
cap = cv2.VideoCapture(0)

# =======================
# PYGAME SETUP
# =======================
pygame.init()
screen = pygame.display.set_mode((200, 200))
pygame.display.set_caption("RC Car Keyboard Control")

running = True
keys = {"w": False, "s": False, "a": False, "d": False}

# =======================
# ML SETUP (TFLITE)
# =======================
MODEL_PATH = "/home/capstone/Desktop/capstone/model/tflite-model"
INPUT_SIZE = (160, 160)
ML_EVERY_N_FRAMES = 10
frame_count = 0
stop_flag = False

# Load TFLite model
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define labels
labels = ["redlight", "greenlight", "stop"]

# =======================
# HELPER FUNCTIONS
# =======================
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

def update_motors():
    if stop_flag:
        stop_motors()
        return

    motor1 = 0
    motor2 = 0

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


# ML inference
def run_traffic_ml(frame):
    # Preprocess frame
    img = cv2.resize(frame, INPUT_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0  # normalize
    input_data = np.expand_dims(img, axis=0)

    # Set tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get output
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    # Get label with max probability
    max_idx = np.argmax(output_data)
    confidence = output_data[max_idx]
    return labels[max_idx], confidence
    
# =======================
# MAIN LOOP
# =======================
while running:
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

    # ================= ML INFERENCE =================
    if frame_count % ML_EVERY_N_FRAMES == 0:
        ret, frame = cap.read()
        if ret:
            label, confidence = run_traffic_ml(frame)
            # Optional: only stop if confidence > 0.8
            if label in ["redlight", "stop"] and confidence > 0.8:
                stop_flag = True
                print(f"? {label.upper()} detected ({confidence:.2f})")
            elif label == "greenlight" and confidence > 0.8:
                stop_flag = False
                print(f"? GREEN detected ({confidence:.2f})")

    frame_count += 1

    # ================= UPDATE MOTORS =================
    update_motors()
    pygame.time.delay(20)

# ================= CLEAN UP =================
stop_motors()
cap.release()
pygame.quit()
