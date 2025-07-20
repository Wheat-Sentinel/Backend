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
from image_processing import process_video_frame

# Global variable to control video capture thread
video_running = False
last_detection_time = 0
from concurrent.futures import ThreadPoolExecutor
thread_pool = ThreadPoolExecutor(max_workers=4)
def send_detection_async(img, disease_name, confidence):
    """
    Wrapper function to process detection in a separate thread
    """
   
    from api_client import send_detection
    
    # Submit the task to the thread pool
    thread_pool.submit(send_detection, img, disease_name, confidence)
    
    return True
def start_video_capture():
    """
    Start capturing video from the camera and process it in real time,
    showing live detections with segmentation masks as polygons.
    """
    global video_running, last_detection_time

    cap = cv2.VideoCapture(config.CAMERA_ID)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.FPS)

    if not cap.isOpened():
        print("ERROR: Could not open video device")
        return

    video_running = True
    print("Video capture started - press 'q' to quit")
    
    while video_running:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Could not read frame")
            break

        # Process the frame using our modular function - directly
        # instead of using thread_pool for this CPU-intensive task
        result_frame, detected_disease, detected_confidence = process_video_frame(frame)
        
        # If a disease was detected with sufficient confidence, send it for processing
        if detected_disease and detected_confidence > 0:
            # Convert result frame to PIL Image for processing
            result_img = Image.fromarray(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB))
            
            print(f"Detected disease: {detected_disease} with {detected_confidence:.2f} confidence")
            
            # Send the detection with the highest confidence
            send_detection_async(result_img, detected_disease, detected_confidence)

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