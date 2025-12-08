from picamera2 import Picamera2
import time
import cv2

# --- ADJUST THESE VALUES UNTIL ROI IS CORRECT ---
x, y = 400, 900        # top-left corner of ROI
w, h = 1600, 1500        # width and height
# -------------------------------------------------

picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration())

picam2.start()
time.sleep(1)

# Capture full image
frame = picam2.capture_array()

# Draw ROI rectangle on the image (green box)
preview = frame.copy()
cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 3)

# Save the image with the drawn ROI box
cv2.imwrite("roi_preview.jpg", preview)
print("Saved roi_preview.jpg ? adjust x,y,w,h until the box is correct.")

picam2.stop()
