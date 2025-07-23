import os
import cv2
from PIL import Image
from image_processing import process_video_frame
import video_capture
def process_video_file(video_path):
    """
    Process a video file for wheat disease detection and send detections to the API.
    
    Args:
        video_path (str): Path to the video file to process
    """
    # Check if file exists
    if not os.path.exists(video_path):
        print(f"ERROR: Video file does not exist: {video_path}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Looking for absolute path: {os.path.abspath(video_path)}")
        return
    
    # Initialize video capture from file
    print(f"Attempting to open video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"ERROR: Could not open video file: {video_path}")
        print("Please check if the video format is supported by OpenCV")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video file loaded successfully: {video_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {frame_count}")
    print("Controls:")
    print("  'q' - Quit")
    print("  'space' - Pause/resume video")
    
    
    delay = int(1000 / fps)  # Convert FPS to milliseconds delay
    
    frame_idx = 0
    detection_count = 0
    paused = False
    
    # Read the first frame to verify video is working
    ret, first_frame = cap.read()
    if not ret:
        print("ERROR: Could not read any frames from the video")
        cap.release()
        return
        
    print(f"First frame read successfully. Starting playback with delay={delay}ms")
    
    # Reset to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    while True:
        # Check for paused state
        if paused:
            # Still show the frame but don't advance
            key = cv2.waitKey(30) & 0xFF
            if key == ord(' '):  # Resume on space
                paused = False
                print("Video resumed")
            elif key == ord('q'):
                break
            continue
        
        ret, frame = cap.read()
        if not ret:
            if frame_idx == 0:
                print("ERROR: Could not read first frame. Video may be corrupt or in an unsupported format.")
                break
            else:
                print("End of video reached")
                
                # Option to loop back to the start
                restart = input("Replay video? (y/n): ")
                if restart.lower() == 'y':
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_idx = 0
                    print("Restarting video from beginning")
                    continue
                else:
                    break

        frame_idx += 1
        
        # Process the frame to detect diseases
        result_frame, detected_disease, detected_confidence = process_video_frame(frame)
        
        # If a disease was detected with sufficient confidence, send it to the API
        if detected_disease and detected_confidence > 0:
            # Convert result frame to PIL Image for processing
            result_img = Image.fromarray(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB))
            
            print(f"Detected disease: {detected_disease} with {detected_confidence:.2f} confidence")
            detection_count += 1
            
            # Send the detection to the API
            video_capture.send_detection_async(result_img, detected_disease, detected_confidence)
        
        # Add frame info and detection count to the frame
        cv2.putText(result_frame, f"Frame: {frame_idx}/{frame_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(result_frame, f"Detections: {detection_count}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show the frame with detections
        cv2.imshow('Wheat Disease Detection - Video File', result_frame)
        
        # Handle key presses
        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # Pause/resume on space
            paused = True
            print("Video paused. Press space to resume.")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print(f"Video processing complete. Sent {detection_count} detections to API.")