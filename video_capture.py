"""
Video capture module for wheat disease detection application.
Handles video capture, processing, and displaying real-time detections.
"""
import cv2
import numpy as np
import threading
import time
from PIL import Image

import config
from image_processing import process_detection_async, model

# Global variable to control video capture thread
video_running = False
last_detection_time = 0

def start_video_capture():
    """
    Start capturing video from the camera and process it in real time,
    showing live detections with segmentation masks as polygons.
    """
    global video_running, last_detection_time

    cap = cv2.VideoCapture(config.CAMERA_ID)

    # Windows DirectShow backend
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.FPS)

    if not cap.isOpened():
        print("ERROR: Could not open video device")
        return

    video_running = True
    print("Video capture started - press 'q' to quit")
    
    # Define colors for different classes - using a colorful palette
    colors = [
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

    while video_running:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Could not read frame")
            break

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
            
            # Process each detection
            for i, detection in enumerate(boxes.data):
                if len(detection) >= 6:  # Ensure we have class, confidence values
                    confidence = float(detection[4])  # Confidence score
                    class_id = int(detection[5])      # Class ID
                    
                    if confidence >= config.CONFIDENCE_THRESHOLD:
                        class_name = results[0].names.get(class_id, f"Class {class_id}")
                        disease_name = f"Wheat {class_name}"
                        color = colors[class_id % len(colors)]  # Assign color based on class ID
                        
                        # Draw bounding box
                        x1, y1, x2, y2 = map(int, detection[:4])
                        cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Add label with class name and confidence
                        label = f"{disease_name}: {confidence:.2f}"
                        cv2.putText(result_frame, label, (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        # Draw segmentation mask if available and we're using a segmentation model
                        if config.IS_SEGMENTATION_MODEL:
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
                            
                        # Process detection if it meets timing criteria
                        current_time = time.time()
                        time_since_last_detection = current_time - last_detection_time
                        
                        if time_since_last_detection >= config.MIN_DETECTION_INTERVAL:
                            last_detection_time = current_time
                            print(f"Processing detection: {disease_name} with {confidence:.2f} confidence")
                            
                            # Convert current frame with boxes and masks to PIL for processing
                            result_img = Image.fromarray(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB))
                            # Use the async version to prevent freezing
                            #process_detection_async(result_img, disease_name, confidence)
                            
                            # Only process one detection per frame to reduce server load
                            break

        # Show result in a window
        cv2.imshow('Wheat Disease Detection - Live', result_frame)

        # Quit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            video_running = False

    cap.release()
    cv2.destroyAllWindows()
    print("Video capture ended.")

def start_video_thread():
    """
    Start the video capture in a separate thread.
    This allows the video processing to run alongside other processes.
    """
    video_thread = threading.Thread(target=start_video_capture)
    video_thread.daemon = True  # Thread will be terminated when main program exits
    video_thread.start()
    return video_thread

def stop_video():
    """
    Stop the video capture thread.
    """
    global video_running
    video_running = False