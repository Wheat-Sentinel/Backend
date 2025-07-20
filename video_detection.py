"""
Wheat disease detection model functionality using YOLO with OpenCV camera input.
Main entry point for the wheat disease detection system.
"""
import gps_module
import video_capture

# Start threads when this module is imported
if __name__ == "__main__":
    # Start the GPS listener thread
    print("Starting GPS listener thread...")
    gps_thread = gps_module.start_gps_thread()
    
    # Start video capture in the main thread
    print("Starting video capture in main thread...")
    video_capture.start_video_capture()

