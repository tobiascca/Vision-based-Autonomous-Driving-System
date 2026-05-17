from picamera2 import Picamera2
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import os

# ----------------------------
# Configuration
# ----------------------------
MODEL_PATH = "/home/capstone/Desktop/capstone/object/tflite_learn_860574_3_v2.tflite"

# Edge Impulse FOMO labels (class 0 must be background)
LABELS = ["background", "stop", "redlight", "greenlight"]

THRESHOLD = 0.8
BOX_EXPAND = 1.0   # ? increase box size (try 1.5?2.5)

# ----------------------------
# Load TFLite model
# ----------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("? Model loaded")
print("Input shape:", input_details[0]["shape"])
print("Output shape:", output_details[0]["shape"])

# ----------------------------
# Camera setup
# ----------------------------
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (320, 240)}
)
picam2.configure(config)
picam2.start()

print("? Camera started")

# ----------------------------
# Main loop
# ----------------------------
try:
    while True:
        frame = picam2.capture_array()

        # ----------------------------
        # Preprocess image
        # ----------------------------
        input_h = input_details[0]["shape"][1]
        input_w = input_details[0]["shape"][2]

        img = cv2.resize(frame, (input_w, input_h))
        img = img.astype(np.float32) / 255.0

        # INT8 quantization
        in_scale, in_zero = input_details[0]["quantization"]
        img = img / in_scale + in_zero
        img = img.astype(np.int8)

        img = np.expand_dims(img, axis=0)

        # ----------------------------
        # Inference
        # ----------------------------
        interpreter.set_tensor(input_details[0]["index"], img)
        interpreter.invoke()

        # ----------------------------
        # Get FOMO output
        # ----------------------------
        output = interpreter.get_tensor(output_details[0]["index"])[0]

        # Dequantize output
        out_scale, out_zero = output_details[0]["quantization"]
        output = (output.astype(np.float32) - out_zero) * out_scale

        # ----------------------------
        # Draw expanded bounding boxes
        # ----------------------------
        h, w, _ = frame.shape
        grid_h, grid_w, num_classes = output.shape

        cell_w = w / grid_w
        cell_h = h / grid_h

        for gy in range(grid_h):
            for gx in range(grid_w):
                class_id = np.argmax(output[gy, gx])
                score = output[gy, gx, class_id]

                if class_id != 0 and score > THRESHOLD:
                    label = LABELS[class_id]

                    # Center of grid cell
                    cx = int((gx + 0.5) * cell_w)
                    cy = int((gy + 0.5) * cell_h)

                    # Expanded box size
                    bw = int(cell_w * BOX_EXPAND)
                    bh = int(cell_h * BOX_EXPAND)

                    x1 = max(0, cx - bw // 2)
                    y1 = max(0, cy - bh // 2)
                    x2 = min(w, cx + bw // 2)
                    y2 = min(h, cy + bh // 2)

                    cv2.rectangle(frame, (x1, y1), (x2, y2),
                                  (0, 255, 0), 2)

                    cv2.putText(frame,
                                f"{label} {score:.2f}",
                                (x1, max(y1 - 5, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                2)

        cv2.imshow("Edge Impulse FOMO Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cv2.destroyAllWindows()
    picam2.close()
    print("? Camera closed")
