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
import image_processing

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
        result_frame, detected_disease, detected_confidence = image_processing.process_video_frame(frame)

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

def start_delayed_video_capture(delay_seconds=2):
    """
    Start video capture with a buffer delay to improve processing performance.
    Always plays at normal speed.
    
    Args:
        delay_seconds (int): Number of seconds to buffer before displaying
    """
    global video_running, last_detection_time
    
    print(f"Starting video capture with {delay_seconds}-second delay buffer...")
    video_running = True
    
    # Initialize the camera
    cap = cv2.VideoCapture(config.CAMERA_ID)
    if not cap.isOpened():
        print("ERROR: Could not open camera")
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.FPS)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_time = 1.0 / actual_fps  # Time per frame in seconds
    
    print(f"Camera initialized at {width}x{height}, {actual_fps} FPS")
    print("Controls:")
    print("  'q' - Quit")
    
    # Create a frame buffer to hold delayed frames with timestamps
    max_buffer_size = int(actual_fps * delay_seconds)  # Buffer size based on actual FPS
    frame_buffer = []
    
    # Fill the buffer first before starting display
    print(f"Building frame buffer... (waiting {delay_seconds} seconds)")
    start_time = time.time()
    
    while time.time() - start_time < delay_seconds and video_running:
        ret, frame = cap.read()
        if ret:
            # Store frame with capture timestamp
            frame_buffer.append((frame.copy(), time.time()))
        else:
            print("WARNING: Failed to read from camera")
        
        # Show a loading indicator
        progress = int(((time.time() - start_time) / delay_seconds) * 100)
        loading_frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(loading_frame, f"Building buffer: {progress}%", 
                   (width//2 - 150, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Wheat Disease Detection - Delayed', loading_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            video_running = False
            cap.release()
            cv2.destroyAllWindows()
            return
    
    print(f"Buffer filled with {len(frame_buffer)} frames. Starting display...")
    frame_count = 0
    detection_count = 0
    
    # For timing control
    next_frame_time = time.time()
    
    # Main processing loop
    while video_running:
        current_time = time.time()
        
        # Capture a new frame and add to buffer
        ret, frame = cap.read()
        if ret:
            # Store frame with capture timestamp
            frame_buffer.append((frame.copy(), current_time))
            
            # Keep buffer at consistent size
            if len(frame_buffer) > max_buffer_size:
                # Get the oldest frame from buffer
                process_frame, frame_timestamp = frame_buffer.pop(0)
                
                # Calculate how much time should pass between frames for natural playback
                time_to_wait = next_frame_time - current_time
                if time_to_wait > 0:
                    time.sleep(time_to_wait)  # Wait the appropriate time to maintain correct speed
                
                # Update next frame time for normal pace playback
                next_frame_time = current_time + frame_time
                
                # Process the frame to detect diseases
                result_frame, detected_disease, detected_confidence = image_processing.process_video_frame(process_frame)
                
                # If a disease was detected with sufficient confidence, send it to the API
                if detected_disease and detected_confidence > 0:
                    # Convert result frame to PIL Image for processing
                    result_img = Image.fromarray(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB))
                    
                    print(f"Detected disease: {detected_disease} with {detected_confidence:.2f} confidence")
                    detection_count += 1
                    
                    # Send the detection to the API
                    send_detection_async(result_img, detected_disease, detected_confidence)
                
                # Add frame info and detection count to the frame
                cv2.putText(result_frame, f"Frame: {frame_count}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(result_frame, f"Detections: {detection_count}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(result_frame, f"Delayed feed ({delay_seconds}s buffer)", 
                           (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                
                # Show the frame with detections
                cv2.imshow('Wheat Disease Detection - Delayed', result_frame)
                
                frame_count += 1
        
        # Check for key presses with a short timeout
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            video_running = False
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print(f"Video capture ended. Processed {frame_count} frames with {detection_count} detections.")

def start_video_thread():
    """
    Start the video capture in a separate thread.
    This allows the video processing to run alongside other processes.
    """
    video_thread = threading.Thread(target=start_video_capture)
    video_thread.daemon = True  # Thread will be terminated when main program exits
    video_thread.start()
    return video_thread

def start_delayed_video_thread(delay_seconds=2):
    """
    Start the delayed video capture in a separate thread.
    
    Args:
        delay_seconds (int): Number of seconds to buffer before displaying
    """
    video_thread = threading.Thread(target=start_delayed_video_capture, args=(delay_seconds,))
    video_thread.daemon = True  # Thread will be terminated when main program exits
    video_thread.start()
    return video_thread

def stop_video():
    """
    Stop the video capture thread.
    """
    global video_running
    video_running = False