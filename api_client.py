"""
API client module for wheat disease detection application.
Handles all communication with backend services, including image uploads.
"""
import datetime
import uuid
from io import BytesIO
import requests
import json
import traceback

import config
import gps_module

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
        client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
        
        # Upload file to Supabase
        response = client.storage.from_(config.STORAGE_BUCKET_NAME).upload(
            path=file_name,
            file=image_bytes,
            file_options={"content-type": "image/jpeg"}
        )
        
        if hasattr(response, 'error') and response.error:
            raise Exception(f"Error uploading to Supabase: {response.error}")
        
        # Generate public URL
        public_url = client.storage.from_(config.STORAGE_BUCKET_NAME).get_public_url(file_name)
        return public_url
    
    except Exception as e:
        print(f"Error uploading to Supabase: {str(e)}")
        raise

def send_detection(img, disease_name, confidence):
    """
    Send detection data to Node.js API when confidence exceeds threshold.
    Upload image to Supabase and send only the image URL to the API.
    
    Args:
        img: The PIL image with detection annotations
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
        file_name = f"public/{config.DEVICE_ID}/{config.ZONE_ID}/{unique_id}.jpg"
        
        # Convert PIL Image to bytes for Supabase upload
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        image_bytes = buffered.getvalue()
        
        # Upload image to Supabase and get URL
        image_url = upload_image(image_bytes, file_name)
        print(f"Image uploaded to Supabase: {image_url}")
        
        # Get GPS data from the GPS module
        gps_data = gps_module.get_full_gps_data()
        
        # Set default coordinates to small non-zero values to avoid falsy validation in JavaScript
        latitude = 0.0001
        longitude = 0.0001
        
        # Try to use real GPS data if available
        if gps_data['latitude'] is not None and gps_data['longitude'] is not None:
            latitude = gps_data['latitude']
            longitude = gps_data['longitude']
            print(f"Using actual GPS coordinates: Lat {latitude}, Long {longitude}")
        else:
            print(f"GPS not available, using default coordinates: Lat {latitude}, Long {longitude}")
        
        cleaned_disease_name = disease_name
        if disease_name.lower().startswith('wheat wheat'):
            cleaned_disease_name = disease_name[6:]
        
        # Prepare simplified payload - always includes default coordinates
        payload = {
            "disease": cleaned_disease_name,
            "confidence": confidence,
            "timestamp": timestamp,
            "zone_id": config.ZONE_ID,
            "device_id": config.DEVICE_ID,
            "image_url": image_url,
            "latitude": latitude,
            "longitude": longitude
        }
        
        print(f"Detection payload ready: {cleaned_disease_name} with {confidence:.2f} confidence")
        
        # Send POST request to Node.js API backend with authentication
        try:
            headers = {
                'Content-Type': 'application/json'
            }
            
            # Add API key if available
            if config.API_KEY:
                headers['x-api-key'] = config.API_KEY
                headers['X-API-Key'] = config.API_KEY
            else:
                print("WARNING: No API key found in environment variables")
            
            # Ensure API URL is valid
            if not config.NODE_API_URL:
                print("ERROR: NODE_API_URL is not set in environment variables")
                return False
                
            DETECTIONS_URL = f"{config.NODE_API_URL.rstrip('/')}/detections" if not config.NODE_API_URL.endswith('/detections') else config.NODE_API_URL
            
            # Send the request
            print(f"Sending request to: {DETECTIONS_URL}")
            print(f"JSON payload: {json.dumps(payload)}")
            
            response = requests.post(DETECTIONS_URL, json=payload, headers=headers, timeout=30)
            
            if response.status_code == 200:
                print(f"Successfully sent detection to API: {cleaned_disease_name} ({confidence:.2f})")
                
                # Send push notification if enabled
                if config.ENABLE_PUSH_NOTIFICATIONS:
                    try:
                        send_disease_detection_notification(cleaned_disease_name, confidence, latitude, longitude)
                    except Exception as e:
                        print(f"Failed to send push notification: {str(e)}")
                
                return True
            else:
                print(f"Failed to send detection to API. Status code: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to API server: {str(e)}")
            return False
            
    except Exception as e:
        print(f"Error processing detection: {str(e)}")
        print(traceback.format_exc())
        return False