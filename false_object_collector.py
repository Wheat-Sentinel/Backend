"""
False object collector for model training data collection.
Opens a webcam stream, detects objects with YOLO, and saves frames with 
detections in YOLO format for negative example training.
"""
import cv2
import numpy as np
import os
import time
import argparse
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from ultralytics import YOLO
from PIL import Image

import config
from image_processing import model

# Directories for saving data
DATASET_DIR = "false_object_dataset"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_DIR = os.path.join(DATASET_DIR, "labels")
CROPS_DIR = os.path.join(DATASET_DIR, "crops")  # Optional cropped detections

# Create directories if they don't exist
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)
os.makedirs(CROPS_DIR, exist_ok=True)

# Class ID for false objects
FALSE_OBJECT_CLASS_ID = 0  # "NotObject" class

# Default confidence threshold
DEFAULT_CONFIDENCE_THRESHOLD = 0.7

# Create a thread pool for I/O operations
save_pool = ThreadPoolExecutor(max_workers=8)

# Queue for pending confirmations
pending_saves = queue.Queue()

def save_yolo_annotation(detections, image_width, image_height, label_path, confidence_threshold):
    """
    Save detections in YOLO format.
    Format: <class_id> <x_center> <y_center> <width> <height>
    All values are normalized to [0, 1]
    Only saves detections above the confidence threshold.
    """
    saved_count = 0
    with open(label_path, 'w') as f:
        for detection in detections:
            if len(detection) >= 6:  # Ensure we have class, confidence values
                confidence = float(detection[4])
                
                # Only save detections above the confidence threshold
                if confidence >= confidence_threshold:
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = detection[:4]
                    
                    # Calculate normalized center coordinates and dimensions
                    x_center = ((x1 + x2) / 2) / image_width
                    y_center = ((y1 + y2) / 2) / image_height
                    width = (x2 - x1) / image_width
                    height = (y2 - y1) / image_height
                    
                    # Always use class 0 for false objects
                    class_id = FALSE_OBJECT_CLASS_ID
                    
                    # Write in YOLO format
                    f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
                    saved_count += 1
    
    return saved_count

def save_cropped_detection(image, detection, crop_path, padding=10):
    """
    Save a cropped image of the detection with padding.
    """
    x1, y1, x2, y2 = map(int, detection[:4])
    
    # Add padding (but ensure within image bounds)
    height, width = image.shape[:2]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(width, x2 + padding)
    y2 = min(height, y2 + padding)
    
    # Crop and save
    crop = image[y1:y2, x1:x2]
    cv2.imwrite(crop_path, crop)

def async_save_data(frame, boxes, confidence_threshold, save_index):
    """
    Asynchronously save image, annotations and crops
    """
    try:
        timestamp = int(time.time())
        image_name = f"{timestamp}_{save_index}.jpg"
        label_name = f"{timestamp}_{save_index}.txt"
        
        # Save paths
        image_path = os.path.join(IMAGES_DIR, image_name)
        label_path = os.path.join(LABELS_DIR, label_name)
        
        # Save the image
        cv2.imwrite(image_path, frame)
        
        # Get image dimensions for normalization
        height, width = frame.shape[:2]
        
        # Save YOLO format annotation
        annotations_saved = save_yolo_annotation(boxes, width, height, label_path, confidence_threshold)
        
        # Save individual crops (optional)
        for i, detection in enumerate(boxes):
            crop_name = f"{timestamp}_{save_index}_crop{i}.jpg"
            crop_path = os.path.join(CROPS_DIR, crop_name)
            save_cropped_detection(frame, detection, crop_path)
            
        print(f"✓ Saved image: {image_path} with {annotations_saved} annotations")
        return True
    except Exception as e:
        print(f"Error saving data: {str(e)}")
        return False

def run_false_object_collector(confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
    """
    Main function to collect false object data.
    
    Args:
        confidence_threshold (float): Minimum confidence score for detections to be saved
    """
    # Initialize camera
    cap = cv2.VideoCapture(config.CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.FPS)

    if not cap.isOpened():
        print("ERROR: Could not open video device")
        return

    print(f"False object collector started - confidence threshold: {confidence_threshold}")
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save current detection")
    print("  '+/-' - Adjust confidence threshold")
    print("  'a' - Toggle auto-save mode (off by default)")
    
    frame_count = 0
    saved_count = 0
    auto_save = False
    pending_save_frame = None
    pending_boxes = None
    save_requested = False
    
    # For tracking save operations
    future_saves = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Could not read frame")
            break

        frame_count += 1
        
        # Only process every 5th frame to avoid too many similar images
        if frame_count % 5 != 0:
            # Show the frame but don't process it
            if pending_save_frame is not None:
                # Keep showing the pending save frame
                display_frame = pending_save_frame.copy()
                cv2.putText(display_frame, "Press 's' to save or any other key to skip", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow('False Object Collector', display_frame)
            else:
                cv2.imshow('False Object Collector', frame)
                
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and pending_save_frame is not None:
                save_requested = True
            elif key != 255 and pending_save_frame is not None:  # Any other key skips the save
                pending_save_frame = None
                pending_boxes = None
                print("Save skipped")
            elif key == ord('a'):
                auto_save = not auto_save
                print(f"Auto-save mode: {'ON' if auto_save else 'OFF'}")
            
            continue
            
        # Convert OpenCV BGR frame to PIL Image (RGB) for YOLO
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Run YOLO model prediction
        results = model.predict(
            source=img,
            show=False,
            imgsz=640,
            verbose=False
        )

        display_frame = frame.copy()
        save_this_frame = False
        
        # Clean up completed save operations
        for future in list(future_saves):
            if future.done():
                future_saves.remove(future)
                
        # Process save request if pending
        if save_requested and pending_save_frame is not None and pending_boxes is not None:
            # Submit save task to thread pool
            future = save_pool.submit(
                async_save_data, 
                pending_save_frame.copy(), 
                pending_boxes.copy(), 
                confidence_threshold, 
                saved_count
            )
            future_saves.append(future)
            saved_count += 1
            pending_save_frame = None
            pending_boxes = None
            save_requested = False

        # Process and save detections
        if results and len(results[0].boxes) > 0:
            # Filter boxes by confidence threshold first
            boxes = results[0].boxes.data
            filtered_boxes = [box for box in boxes if len(box) >= 5 and float(box[4]) >= confidence_threshold]
            
            if filtered_boxes:
                # If auto-save is on, save without prompting
                if auto_save:
                    # Submit save task to thread pool
                    future = save_pool.submit(
                        async_save_data, 
                        frame.copy(), 
                        filtered_boxes.copy(), 
                        confidence_threshold, 
                        saved_count
                    )
                    future_saves.append(future)
                    saved_count += 1
                else:
                    # Store frame for potential saving
                    pending_save_frame = frame.copy()
                    pending_boxes = filtered_boxes.copy()
                    
                # Draw boxes on the display frame for all detections
                for box in boxes:
                    if len(box) >= 5:
                        x1, y1, x2, y2 = map(int, box[:4])
                        conf = float(box[4])
                        
                        # Use different colors based on threshold
                        if conf >= confidence_threshold:
                            color = (0, 0, 255)  # Red for detections meeting threshold
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(display_frame, f"NotObject: {conf:.2f}", 
                                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        else:
                            # Draw boxes that don't meet the threshold with a different color
                            color = (0, 165, 255)  # Orange for below threshold
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 1)  # Thinner line
                            cv2.putText(display_frame, f"Low conf: {conf:.2f}", 
                                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # Add save prompt if not in auto-save mode
                if not auto_save and pending_save_frame is not None:
                    cv2.putText(display_frame, "Press 's' to save or any other key to skip", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                pending_save_frame = None
                pending_boxes = None
        else:
            pending_save_frame = None
            pending_boxes = None
            
        # Add counter and threshold to the frame
        cv2.putText(display_frame, f"Saved: {saved_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Threshold: {confidence_threshold}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Show auto-save status
        auto_save_text = f"Auto-save: {'ON' if auto_save else 'OFF'}"
        cv2.putText(display_frame, auto_save_text, (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if auto_save else (0, 0, 255), 2)
                   
        # Show active saving tasks
        if future_saves:
            cv2.putText(display_frame, f"Saving: {len(future_saves)} tasks", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        # Show the frame with detections
        cv2.imshow('False Object Collector', display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        # Allow adjusting threshold with + and - keys
        elif key == ord('+') or key == ord('='):
            confidence_threshold = min(1.0, confidence_threshold + 0.05)
            print(f"Increased confidence threshold to: {confidence_threshold:.2f}")
        elif key == ord('-') or key == ord('_'):
            confidence_threshold = max(0.0, confidence_threshold - 0.05)
            print(f"Decreased confidence threshold to: {confidence_threshold:.2f}")
        elif key == ord('s') and pending_save_frame is not None:
            save_requested = True
        elif key == ord('a'):
            auto_save = not auto_save
            print(f"Auto-save mode: {'ON' if auto_save else 'OFF'}")

    # Wait for any pending save operations to complete
    print("Waiting for pending save operations to complete...")
    save_pool.shutdown(wait=True)
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print(f"Collection complete. Saved {saved_count} images with annotations.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect false object training data")
    parser.add_argument("--confidence", "-c", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD,
                        help=f"Confidence threshold (0.0-1.0, default: {DEFAULT_CONFIDENCE_THRESHOLD})")
    args = parser.parse_args()
    
    run_false_object_collector(args.confidence)