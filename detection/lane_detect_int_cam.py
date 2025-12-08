import cv2
import numpy as np

# Reduce resolution for Pi performance
FRAME_W, FRAME_H = 320, 240

def region_of_interest(img):
    h = img.shape[0]
    polygon = np.array([[
        (0, h),
        (0, int(h*0.55)),
        (img.shape[1], int(h*0.55)),
        (img.shape[1], h)
    ]])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(img, mask)

def detect_lanes(frame):
    # Resize for speed
    frame_small = cv2.resize(frame, (FRAME_W, FRAME_H))

    # Convert to grayscale
    gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

    # Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny edge detector
    edges = cv2.Canny(blur, 60, 150)

    # Focus on bottom half of the frame
    cropped_edges = region_of_interest(edges)

    # Hough Transform
    lines = cv2.HoughLinesP(
        cropped_edges,
        rho=1,
        theta=np.pi/180,
        threshold=50,
        minLineLength=40,
        maxLineGap=100
    )

    # Draw lines on original (resized) frame
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame_small, (x1, y1), (x2, y2), (0, 255, 0), 3)

    return frame_small


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, 30)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed = detect_lanes(frame)

        cv2.imshow("Live Lane Detection", processed)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
