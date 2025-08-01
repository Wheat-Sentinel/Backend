"""
Image processing module for wheat disease detection application.
Handles image analysis, disease detection using YOLO with ByteTracker, and API communication.
"""
from ultralytics import YOLO
from PIL import Image
import time
import cv2
import numpy as np
import torch
import supervision as sv
from concurrent.futures import ThreadPoolExecutor

import config

# Global variables for ByteTracker
sent_track_ids = set()  # Set to track which track IDs have been sent to API

# Thread pool for async API calls
thread_pool = ThreadPoolExecutor(max_workers=4)

# Check if CUDA is available and set device accordingly
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# Load the model to the appropriate device
model = YOLO(config.MODEL_PATH)

# Initialize ByteTracker using Supervision
tracker = sv.ByteTrack(
    track_activation_threshold=0.25,
    lost_track_buffer=30,
    frame_rate=30
)

box_annotator = sv.RoundBoxAnnotator(thickness=2, color_lookup=sv.ColorLookup.CLASS)
label_annotator = sv.LabelAnnotator(text_color=sv.Color.WHITE, text_scale=1, text_thickness=2)

# For segmentation masks (if using segmentation model)
mask_annotator = sv.MaskAnnotator(
    opacity=0.3  # Semi-transparent overlay
)



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


def send_detection_async(img, disease_name, confidence):
    """
    Send detection to API asynchronously.
    
    Args:
        img (PIL.Image): The processed frame as PIL Image
        disease_name (str): Name of the detected disease
        confidence (float): Confidence score of the detection
    """
    try:
        from api_client import send_detection
        # Submit the task to the thread pool
        thread_pool.submit(send_detection, img, disease_name, confidence)
        return True
    except ImportError:
        print("Warning: Could not import api_client, detection not sent to backend")
        return False
    except Exception as e:
        print(f"Error sending detection to backend: {e}")
        return False


def process_video_frame(frame):
    """
    Process a video frame for wheat disease detection with segmentation masks and ByteTracker.
    Automatically sends new detections to the API asynchronously.
    
    Args:
        frame (numpy.ndarray): The input OpenCV frame to analyze
        
    Returns:
        tuple: (processed_frame) where:
            - processed_frame: The annotated frame with detection results
    """
    global tracker, sent_track_ids
    
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

    
    # Only process detections that meet the threshold
    if results and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        # Only try to get masks if we're using a segmentation model
        masks = results[0].masks if config.IS_SEGMENTATION_MODEL and hasattr(results[0], 'masks') and results[0].masks is not None else None
        
        # Convert YOLO results to Supervision Detections format
        detections = sv.Detections.from_ultralytics(results[0])
        
        # Filter detections by confidence threshold and exclude "notobject" class
        confidence_mask = detections.confidence >= config.CONFIDENCE_THRESHOLD
        
        # Filter out "notobject" class if it exists
        class_mask = np.ones(len(detections), dtype=bool)
        if hasattr(results[0], 'names'):
            for i, class_id in enumerate(detections.class_id):
                class_name = results[0].names.get(class_id, f"Class {class_id}")
                if class_name.lower() == "notobject":
                    class_mask[i] = False
        
        # Apply both filters
        final_mask = confidence_mask & class_mask
        filtered_detections = detections[final_mask]
        
        # Update tracker with filtered detections
        tracks = tracker.update_with_detections(filtered_detections)
        
        highest_confidence = 0.0
        highest_confidence_disease = None # List to store objects that should be sent to backend
        
        # Draw all tracked detections for visualization using Supervision annotators
        if len(tracks) > 0:
            # Create labels for each track
            labels = []
            
            for i in range(len(tracks)):
                if tracks.tracker_id[i] is None:  # Skip untracked detections
                    labels.append("")  # Add empty label for untracked detections
                    continue
                    
                track_id = tracks.tracker_id[i]
                confidence = tracks.confidence[i]
                class_id = tracks.class_id[i]
                class_name = results[0].names.get(class_id, f"Class {class_id}")
                
                disease_name = class_name
                
                # Create label with class name, confidence, and track ID
                label = f"ID:{int(track_id)} {disease_name}: {float(confidence):.2f}"
                labels.append(label)
            
            # Apply annotations using Supervision with default color scheme
            # Draw bounding boxes
            result_frame = box_annotator.annotate(
                scene=result_frame,
                detections=tracks
            )
            
            # Draw labels
            result_frame = label_annotator.annotate(
                scene=result_frame,
                detections=tracks,
                labels=labels
            )
            
            # Draw segmentation masks if available and we're using a segmentation model
            if config.IS_SEGMENTATION_MODEL and masks is not None:
                # Create a detections object with masks for annotation
                detections_with_masks = sv.Detections(
                    xyxy=tracks.xyxy,
                    confidence=tracks.confidence,
                    class_id=tracks.class_id,
                    tracker_id=tracks.tracker_id,
                    mask=filtered_detections.mask if hasattr(filtered_detections, 'mask') and filtered_detections.mask is not None else None
                )
                
                # Apply mask annotation if masks are available
                if detections_with_masks.mask is not None:
                    result_frame = mask_annotator.annotate(
                        scene=result_frame,
                        detections=detections_with_masks
                    )
        
        # Send new detections to API with annotated frame
        for i in range(len(tracks)):
            if tracks.tracker_id[i] is None:  # Skip untracked detections
                continue
                
            track_id = tracks.tracker_id[i]
            confidence = tracks.confidence[i]
            class_id = tracks.class_id[i]
            class_name = results[0].names.get(class_id, f"Class {class_id}")
            
            # Check if this track ID is new and should be sent to API
            if track_id not in sent_track_ids:
                sent_track_ids.add(track_id)
                
                # Send detection to API asynchronously with annotated frame
                disease_name = class_name
                print(f"Processing new detection: Track ID:{int(track_id)} {disease_name} with {float(confidence):.2f} confidence")
                
                # Convert annotated frame to PIL Image for API
                annotated_img = Image.fromarray(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB))
                send_detection_async(annotated_img, disease_name, float(confidence))
        
       
    
    return result_frame