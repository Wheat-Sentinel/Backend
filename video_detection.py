"""
Wheat disease detection model functionality using YOLO with OpenCV camera input or video file.
Main entry point for the wheat disease detection system.
"""

import argparse
import gps_module
import video_capture
from image_processing import process_video_frame
from api_client import send_detection
from video_prerecorded import process_video_file


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Wheat disease detection system")
    parser.add_argument("--mode", "-m", type=str, choices=["live", "delayed", "video"], default="live",
                        help="Input mode: 'live' for camera, 'delayed' for buffered camera, or 'video' for video file")
    parser.add_argument("--video", "-v", type=str, help="Path to video file (required if mode is 'video')")
    parser.add_argument("--delay", "-d", type=int, default=2, help="Buffer delay in seconds for delayed mode")
    
    args = parser.parse_args()
    
    # Start the GPS listener thread regardless of mode
    print("Starting GPS listener thread...")
    gps_thread = gps_module.start_gps_thread()
    
    if args.mode == "video":
        # Process a video file
        if not args.video:
            print("ERROR: Video path is required when using video mode. Use --video parameter.")
            exit(1)
        print("Processing video file for wheat disease detection...")
        process_video_file(args.video)
    elif args.mode == "delayed":
        # Start delayed video capture with specified buffer
        print(f"Starting delayed video capture with {args.delay}s buffer...")
        video_capture.start_delayed_video_capture(args.delay)
    else:
        # Start regular live video capture
        print("Starting real-time video capture in main thread...")
        video_capture.start_video_capture()

