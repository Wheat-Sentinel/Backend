from ultralytics import YOLO
import config

# Load your YOLO model (.pt)
model = YOLO("models/finetune_detection.pt")  # Adjust path as needed

print(f"Model info:")
print(f"- Classes: {model.names}")
print(f"- Number of classes: {len(model.names)}")

model.export(
    format="onnx",
    imgsz=640,
    simplify=False,
    opset=12,
    nms=True,            # includes non-max suppression
    dynamic=False,
    batch=1              # match inference pipeline
)


print("✅ Export complete: detection.onnx created")
print("Model exported with:")
print(f"- Input size: 640x640")
print(f"- Classes: {len(model.names)}")
print(f"- Output format: YOLOv8/v11 format [1, 4+classes, detections]")
