from picamera2 import Picamera2
import time

# Initialize the camera
picam2 = Picamera2()

# Configure the camera for a still image (e.g., 1080p resolution)
config = picam2.create_still_configuration(main={"size": (1920, 1080)})
picam2.configure(config)

# Start the camera preview/capture process
picam2.start()

# Give the camera a couple of seconds to set its light levels/gain
time.sleep(2) 

# Capture and save the image
picam2.capture_file("st.jpg")

print("Image saved as my_first_photo.jpg")

# Stop the camera
picam2.stop()
