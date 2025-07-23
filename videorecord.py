import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
import argparse
from pathlib import Path


def process_video_with_yolo(source_path, output_path, model_path, conf_threshold=0.3, iou_threshold=0.7):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID', 'MJPG'
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not video_writer.isOpened():
        raise RuntimeError("Cannot open VideoWriter, check codec and file path")

    box_annotator = sv.RoundBoxAnnotator(thickness=2, color_lookup=sv.ColorLookup.CLASS)
    label_annotator = sv.LabelAnnotator(text_color=sv.Color.WHITE, text_scale=1, text_thickness=2)

    for frame in sv.get_video_frames_generator(source_path):
        results = model(frame, conf=conf_threshold, iou=iou_threshold)[0]
        detections = sv.Detections.from_ultralytics(results)

        labels = [
    f"{model.model.names[c]} {conf:.2f}"
    for _, conf, c in zip(detections.xyxy, detections.confidence, detections.class_id)]
        annotated = box_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
        video_writer.write(annotated)

    video_writer.release()
    print(f"Video processed and saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a video with YOLO object detection")
    parser.add_argument("--source", type=str, required=True, help="Path to the source video file")
    parser.add_argument("--output", type=str, default="output.mp4", help="Path for the output video file")
    parser.add_argument("--model", type=str, default="models/detection.pt", help="Path to the YOLO model file")
    parser.add_argument("--conf", type=float, default=0.7, help="Confidence threshold for detections")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for NMS")
    
    args = parser.parse_args()
    
    process_video_with_yolo(
        source_path=args.source,
        output_path=args.output,
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )