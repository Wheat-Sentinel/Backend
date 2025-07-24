"""
Image processing module for wheat disease detection application.
Handles image analysis and disease detection using YOLO.
"""
from ultralytics import YOLO
from PIL import Image
import time
import cv2
import numpy as np
import torch

import config

# Global variables
last_detection_time = 0

# Check if CUDA is available and set device accordingly
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# Load the model to the appropriate device
model = YOLO(config.MODEL_PATH)


# Define colors for different classes - using a colorful palette
COLORS = [
    (255, 0, 0),    # Blue
    (0, 255, 0),    # Green
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 0, 0),    # Dark blue
    (0, 128, 0),    # Dark green
    (0, 0, 128)     # Dark red
]


def process_video_frame(frame):
    """
    Process a video frame for wheat disease detection with segmentation masks.
    
    Args:
        frame (numpy.ndarray): The input OpenCV frame to analyze
        
    Returns:
        tuple: (processed_frame, detected_disease, confidence) where:
            - processed_frame: The annotated frame with detection results
            - detected_disease: The name of the detected disease with highest confidence or None
            - confidence: The confidence score of the detection or 0.0
    """
    global last_detection_time
    
    # Convert OpenCV BGR frame to PIL Image (RGB)
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Run YOLO model prediction with segmentation masks
    results = model.predict(
        source=img,
        show_labels=False,  # We'll handle labels manually
        show_conf=False,    # We'll handle confidence display manually
        imgsz=640,
        verbose=False,      # Silence the print messages
        retina_masks=True   # Get high quality segmentation masks
    )
    
    # Start with the original frame
    result_frame = frame.copy()
    detected_disease = None
    detected_confidence = 0.0
    
    # Only process detections that meet the threshold
    if results and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        # Only try to get masks if we're using a segmentation model
        masks = results[0].masks if config.IS_SEGMENTATION_MODEL and hasattr(results[0], 'masks') and results[0].masks is not None else None
        
        highest_confidence = 0.0
        highest_confidence_disease = None
        highest_confidence_index = -1
        
        # Process each detection and find the one with highest confidence
        for i, detection in enumerate(boxes.data):
            if len(detection) >= 6:  # Ensure we have class, confidence values
                confidence = float(detection[4])  # Confidence score
                class_id = int(detection[5])      # Class ID
                class_name = results[0].names.get(class_id, f"Class {class_id}")
                
                # Skip "notobject" class predictions
                if class_name.lower() == "notobject":
                    continue
                
                if confidence >= config.CONFIDENCE_THRESHOLD and confidence > highest_confidence:
                    highest_confidence = confidence
                    highest_confidence_disease = class_name
                    highest_confidence_index = i
        
        # Draw all detections for visualization but only return the highest confidence one
        for i, detection in enumerate(boxes.data):
            if len(detection) >= 6:  # Ensure we have class, confidence values
                confidence = float(detection[4])  # Confidence score
                class_id = int(detection[5])      # Class ID
                class_name = results[0].names.get(class_id, f"Class {class_id}")
                
                # Skip "notobject" class predictions
                if class_name.lower() == "notobject":
                    continue
                
                if confidence >= config.CONFIDENCE_THRESHOLD:
                    disease_name = f"Wheat {class_name}"
                    color = COLORS[class_id % len(COLORS)]  # Assign color based on class ID
                    
                    # Draw bounding box
                    x1, y1, x2, y2 = map(int, detection[:4])
                    cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Add label with class name and confidence
                    label = f"{disease_name}: {confidence:.2f}"
                    cv2.putText(result_frame, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Draw segmentation mask if available and we're using a segmentation model
                    if config.IS_SEGMENTATION_MODEL and masks is not None:
                        # Get the segmentation mask for this detection
                        mask = masks[i].data
                        
                        # Convert mask to numpy array for OpenCV
                        mask = mask.cpu().numpy() if hasattr(mask, 'cpu') else mask
                        
                        if mask.ndim == 3:
                            # Some models might output multiple masks per detection
                            mask = mask[0]
                            
                        # Resize mask to frame size
                        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                        
                        # Apply threshold to make a binary mask
                        mask = (mask > 0.5).astype(np.uint8)
                        
                        # Create a properly shaped colored mask
                        colored_mask = np.zeros_like(frame)
                        # Apply the color to each channel separately
                        colored_mask[:,:,0][mask == 1] = color[0]  # B
                        colored_mask[:,:,1][mask == 1] = color[1]  # G
                        colored_mask[:,:,2][mask == 1] = color[2]  # R
                        
                        # Find contours (polygon shapes) from the mask
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        # Draw all contours on the image
                        cv2.drawContours(result_frame, contours, -1, color, 2)
                        
                        # Apply a semi-transparent colored overlay for the segmentation
                        overlay = result_frame.copy()
                        for contour in contours:
                            cv2.fillPoly(overlay, [contour], color)
                        
                        # Blend the overlay with the result frame
                        alpha = 0.3  # Transparency factor
                        cv2.addWeighted(overlay, alpha, result_frame, 1 - alpha, 0, result_frame)
        
        # Check if we should process this detection based on timing
        current_time = time.time()
        time_since_last_detection = current_time - last_detection_time
        
        if highest_confidence_disease and time_since_last_detection >= config.MIN_DETECTION_INTERVAL:
            last_detection_time = current_time
            detected_disease = f"Wheat {highest_confidence_disease}"
            detected_confidence = highest_confidence
            print(f"Processing detection: {detected_disease} with {detected_confidence:.2f} confidence")
    
    return result_frame, detected_disease, detected_confidence