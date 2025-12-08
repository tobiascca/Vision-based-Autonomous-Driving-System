from picamera2 import Picamera2
import cv2
import numpy as np
import math

FRAME_W, FRAME_H = 320, 240

def region_of_interest(img):
    h = img.shape[0]
    polygon = np.array([[
        (0, h),
        (0, int(h * 0.55)),
        (img.shape[1], int(h * 0.55)),
        (img.shape[1], h)
    ]])
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

    lane_mid_x = FRAME_W // 2
    direction_offset =_
