"""
Wheat disease detection model functionality using YOLO.
"""
from ultralytics import YOLO
from PIL import Image
import datetime
import uuid
from io import BytesIO
import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get configuration from environment variables
MODEL_PATH = os.getenv('MODEL_PATH')
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.7'))
DEVICE_ID = os.getenv('DEVICE_ID')
ZONE_ID = os.getenv('ZONE_ID')
NODE_API_URL = os.getenv('NODE_API_URL')

# Print the API URL being used (for debugging)
print(f"API URL from .env file: {NODE_API_URL}")

# Force use of specific URL if needed
# NODE_API_URL = "https://wheat-disease-detection-o7su9etoh-seifeldin-amrs-projects.vercel.app/detections"

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
STORAGE_BUCKET_NAME = os.getenv('STORAGE_BUCKET_NAME')
API_KEY = os.getenv('API_KEY')  # Only use API key for authentication

# Initialize YOLO model
model = YOLO(MODEL_PATH)

def upload_image(image_bytes, file_name):
    """
    Upload image to Supabase storage and return the public URL.
    
    Args:
        image_bytes (bytes): The image as bytes
        file_name (str): The name to save the file as
    
    Returns:
        str: The public URL of the uploaded file
    """
    try:
        import supabase
        from supabase import create_client

        # Create Supabase client
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Upload file to Supabase
        response = client.storage.from_(STORAGE_BUCKET_NAME).upload(
            path=file_name,
            file=image_bytes,
            file_options={"content-type": "image/jpeg"}
        )
        
        if hasattr(response, 'error') and response.error:
            raise Exception(f"Error uploading to Supabase: {response.error}")
        
        # Generate public URL
        public_url = client.storage.from_(STORAGE_BUCKET_NAME).get_public_url(file_name)
        return public_url
    
    except Exception as e:
        print(f"Error uploading to Supabase: {str(e)}")
        raise

def process_detection(img: Image.Image, disease_name: str, confidence: float):
    """
    Send detection data to Node.js API when confidence exceeds threshold.
    Upload image to Supabase and send only the image URL to the API.
    
    Args:
        img (Image.Image): The PIL image with detection annotations
        disease_name (str): Name of the detected disease
        confidence (float): Confidence score of the detection
    
    Returns:
        bool: True if the detection was successfully processed, False otherwise
    """
    try:
        # Current timestamp
        timestamp = datetime.datetime.now().isoformat()
        
        # Generate unique filename for Supabase storage with public directory
        unique_id = str(uuid.uuid4())
        file_name = f"public/{DEVICE_ID}/{ZONE_ID}/{unique_id}.jpg"
        
        # Convert PIL Image to bytes for Supabase upload
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        image_bytes = buffered.getvalue()
        
        # Upload image to Supabase and get URL
        image_url = upload_image(image_bytes, file_name)
        print(f"Image uploaded to Supabase: {image_url}")
        
        # Prepare payload with image URL instead of base64 image
        payload = {
            "disease": disease_name,
            "confidence": confidence,
            "timestamp": timestamp,
            "zone_id": ZONE_ID,
            "device_id": DEVICE_ID,
            "image_url": image_url  # Send the URL instead of base64 image
        }
        
        # Send POST request to Node.js API backend with authentication
        try:
            headers = {
                'Content-Type': 'application/json'
            }
            
            # Check if API key is available and add to headers
            if API_KEY:
                # Add API key in multiple header formats to ensure compatibility
                headers['x-api-key'] = API_KEY
                headers['X-API-Key'] = API_KEY
                print(f"Using API key for authentication: {API_KEY[:5]}...{API_KEY[-5:]} (redacted middle)")
                print(f"API URL being used: {NODE_API_URL}")
                print(f"Headers being sent: {list(headers.keys())}")
            else:
                print("WARNING: No API key found. Authentication will fail!")
                print("Make sure API_KEY is set in your .env file")
            
            # Send the request with headers
            print(f"Sending request to: {NODE_API_URL}")
            
            DETECTIONS_URL = f"{NODE_API_URL.rstrip('/')}/detections"
            response = requests.post(DETECTIONS_URL, json=payload, headers=headers)
            
            
            if response.status_code == 200:
                print(f"Successfully sent detection to Node.js API: {disease_name} ({confidence:.2f})")
                return True
            else:
                print(f"Failed to send detection to Node.js API. Status code: {response.status_code}")
                print(f"Response: {response.text}")
                if response.status_code == 401:
                    print("Authentication error: API key was not accepted")
                    print("Verify that your API_KEY in .env matches the one in your Vercel environment")
                return False
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to Node.js API server: {str(e)}")
            print("Detection not saved. Node.js API server must be running to save detections.")
            return False
            
    except Exception as e:
        print(f"Error processing detection: {str(e)}")
        return False

def detect_wheat_disease(img: Image.Image) -> Image.Image:
    """
    Run YOLO inference on the uploaded image, plotting disease detections.
    
    Args:
        img (Image.Image): The input PIL image to analyze
        
    Returns:
        Image.Image: The annotated image with detection results
    """
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
                
                if confidence >= CONFIDENCE_THRESHOLD:
                    # Get class name (default to class_id if names not available)
                    class_name = results[0].names.get(class_id, f"Class {class_id}")
                    disease_name = f"Wheat {class_name}"
                    
                    # Send high-confidence detection to backend with annotated image
                    process_detection(result_img, disease_name, confidence)
                    
        return result_img
    else:
        return img