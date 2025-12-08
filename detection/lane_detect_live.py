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
    direction_offset = 0

    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Estimate lane center offset (simple)
            lane_mid_x = int((x1 + x2) / 2)
            
        direction_offset = lane_mid_x - (FRAME_W // 2)

    # Draw projected middle line
    cv2.line(
        overlay,
        (FRAME_W // 2, FRAME_H),
        (FRAME_W // 2, FRAME_H - 50),
        (0, 0, 255),
        3
    )

    # Draw direction arrow (steering)
    arrow_length = 40
    arrow_x = FRAME_W // 2 + int(direction_offset * 0.5)
    cv2.arrowedLine(
        overlay,
        (FRAME_W // 2, FRAME_H - 10),
        (arrow_x, FRAME_H - 60),
        (255, 0, 0),
        3,
        tipLength=0.5
    )

    return edges, overlay


def main():
    picam = Picamera2()
    config = picam.create_preview_configuration(
        main={"size": (FRAME_W, FRAME_H), "format": "RGB888"}
    )
    picam.configure(config)
    picam.start()

    while True:
        frame = picam.capture_array()

        # Make copies for displaying
        raw_display = frame.copy()

        # Lane detection
        edges, overlay = detect_lanes(frame)

        # Show windows
        cv2.imshow("Raw Input", raw_display)
        cv2.imshow("Edges / Lane Mask", edges)
        cv2.imshow("Overlay", overlay)

        if cv2.waitKey(1) == 27:  # ESC
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
