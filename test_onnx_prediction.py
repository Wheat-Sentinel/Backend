from ultralytics import YOLO
import cv2

# Load the ONNX model
model = YOLO("models/detection.onnx")  # Make sure this is exported with NMS support

# Run inference
results = model("C:\\Users\\Asus\\Downloads\\0458.png")

# Plot the first result and convert to OpenCV format
result_image = results[0].plot()  # This returns an annotated image (numpy array)

# Show with OpenCV
cv2.imshow("Detection", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
