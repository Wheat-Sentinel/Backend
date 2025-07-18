"""
Image processing module for wheat disease detection application.
Handles image analysis and disease detection using YOLO.
"""
from ultralytics import YOLO
from PIL import Image
import time
from concurrent.futures import ThreadPoolExecutor

import config

# Global variables
last_detection_time = 0
model = YOLO(config.MODEL_PATH)
thread_pool = ThreadPoolExecutor(max_workers=4)

def detect_wheat_disease(img: Image.Image) -> Image.Image:
    """
    Run YOLO inference on the uploaded image, plotting disease detections.
    
    Args:
        img (Image.Image): The input PIL image to analyze
        
    Returns:
        Image.Image: The annotated image with detection results
    """
    global last_detection_time
    
    results = model.predict(
        source=img,
        show_labels=True,
        show_conf=True,
        imgsz=640
    )

    if results:
        # Get annotated image
        annotated = results[0].plot()
        result_img = Image.fromarray(annotated[..., ::-1])
        
        # Check for high-confidence detections
        for detection in results[0].boxes.data:
            if len(detection) >= 6:  # Ensure we have class, confidence values
                confidence = float(detection[4])  # Confidence score
                class_id = int(detection[5])      # Class ID
                
                if confidence >= config.CONFIDENCE_THRESHOLD:
                    # Get class name (default to class_id if names not available)
                    class_name = results[0].names.get(class_id, f"Class {class_id}")
                    disease_name = f"Wheat {class_name}"
                    
                    # Throttle detection processing regardless of disease type
                    current_time = time.time()
                    time_since_last_detection = current_time - last_detection_time
                    
                    if time_since_last_detection < config.MIN_DETECTION_INTERVAL:
                        print(f"Detection suppressed: only {time_since_last_detection:.1f}s since last detection (minimum interval: {config.MIN_DETECTION_INTERVAL}s)")
                        continue  # Skip this detection, too soon since last detection
                    
                    # Update the last detection time
                    last_detection_time = current_time
                    print(f"Processing detection: {disease_name} with {confidence:.2f} confidence")
                    
                    # Import here to avoid circular imports
                    from api_client import process_detection
                    
                    # Send high-confidence detection to backend with annotated image
                    process_detection(result_img, disease_name, confidence)
                    
                    # Only process one detection per frame to reduce server load
                    break
                    
        return result_img
    else:
        return img

def process_detection_async(img, disease_name, confidence):
    """
    Wrapper function to process detection in a separate thread
    """
    # Import here to avoid circular imports
    from api_client import process_detection
    
    # Submit the task to the thread pool
    thread_pool.submit(process_detection, img, disease_name, confidence)
    
    # Return immediately to avoid blocking the video capture loop
    return True