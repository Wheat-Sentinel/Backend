"""
Configuration module for wheat disease detection application.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Model configuration
MODEL_PATH = os.getenv('MODEL_PATH')
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.7'))

# Check if the model is a segmentation model based on the filename
IS_SEGMENTATION_MODEL = False
if MODEL_PATH:
    if 'segementation.pt' in MODEL_PATH or 'segmentation.pt' in MODEL_PATH or 'segment' in MODEL_PATH.lower():
        IS_SEGMENTATION_MODEL = True

# Debug print to help identify the issue
print(f"Model path: {MODEL_PATH}")
print(f"Is segmentation model: {IS_SEGMENTATION_MODEL}")

# Device and location configuration
DEVICE_ID = os.getenv('DEVICE_ID')
ZONE_ID = os.getenv('ZONE_ID')

# API configuration
NODE_API_URL = os.getenv('NODE_API_URL')
API_KEY = os.getenv('API_KEY')  # Only use API key for authentication

# Supabase configuration
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
STORAGE_BUCKET_NAME = os.getenv('STORAGE_BUCKET_NAME')

# Feature flags
ENABLE_PUSH_NOTIFICATIONS = os.getenv('ENABLE_PUSH_NOTIFICATIONS', 'true').lower() == 'true'

# Video capture configuration
CAMERA_ID = 1  
FRAME_WIDTH = int(os.getenv('FRAME_WIDTH', '640'))
FRAME_HEIGHT = int(os.getenv('FRAME_HEIGHT', '480'))
FPS = int(os.getenv('FPS', '30'))

# Detection configuration
MIN_DETECTION_INTERVAL = 10  # 10 seconds between any disease detections

# Print the API URL being used (for debugging)
print(f"API URL from .env file: {NODE_API_URL}")