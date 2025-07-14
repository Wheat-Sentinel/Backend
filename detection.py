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
import socket as soc
import threading
import time
import re
import json
import gradio as gr
from dotenv import load_dotenv
from expo_notifications import send_disease_detection_notification

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
ENABLE_PUSH_NOTIFICATIONS = os.getenv('ENABLE_PUSH_NOTIFICATIONS', 'true').lower() == 'true'

# Global variables to store the latest GPS coordinates
latest_latitude = None
latest_longitude = None
latest_altitude = None
latest_accuracy = None
latest_timestamp = None
gps_lock = threading.Lock()

# Initialize YOLO model
model = YOLO(MODEL_PATH)

# GPS FUNCTIONALITY
def get_local_ip():
    """
    Get the local IP address of the current machine.
    Works even when behind NAT or on Wi-Fi.
    """
    try:
        s = soc.socket(soc.AF_INET, soc.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # Doesn't send packets, just determines local IP
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return '127.0.0.1'  # fallback

def parse_gps_data(data):
    """
    Parse GPS data from the received string.
    Handles both JSON formatted data and simple string patterns.
    Returns tuple of (latitude, longitude) or None if parsing fails.
    """
    global latest_latitude, latest_longitude, latest_altitude, latest_accuracy, latest_timestamp
    
    try:
        # Try to parse as JSON first
        try:
            # Try to convert string representation of dict to actual dict if needed
            if isinstance(data, str) and data.strip().startswith('{') and data.strip().endswith('}'):
                try:
                    # First try standard JSON parsing
                    gps_data = json.loads(data)
                except json.JSONDecodeError:
                    # If that fails, try safer eval for Python dict-like strings
                    import ast
                    gps_data = ast.literal_eval(data)
            else:
                # Already a dict or not in expected format
                gps_data = data if not isinstance(data, str) else json.loads(data)
            
            # Check for the structure in the example: {'fused': {...}, 'network': {...}}
            if isinstance(gps_data, dict) and 'fused' in gps_data and 'latitude' in gps_data['fused'] and 'longitude' in gps_data['fused']:
                lat = float(gps_data['fused']['latitude'])
                lon = float(gps_data['fused']['longitude'])
                alt = float(gps_data['fused'].get('altitude', 0))
                acc = float(gps_data['fused'].get('accuracy', 0))
                timestamp = gps_data['fused'].get('time', int(time.time() * 1000))
                
                # Store all the available data
                with gps_lock:
                    latest_latitude = lat
                    latest_longitude = lon
                    latest_altitude = alt
                    latest_accuracy = acc
                    latest_timestamp = timestamp
                
                print(f"Successfully parsed GPS data: Lat {lat}, Long {lon}, Alt {alt}m, Acc {acc}m")
                return (lat, lon)
            
            return None
        except (json.JSONDecodeError, ValueError, SyntaxError) as e:
            print(f"JSON parsing error: {e}")
            pass
            
        # Fall back to regex patterns if not valid JSON
        # This regex pattern looks for common GPS coordinate formats
        pattern = r'lat[itude]*[:\s=]*(-?\d+\.?\d*)[,\s]+lon[gitude]*[:\s=]*(-?\d+\.?\d*)'
        match = re.search(pattern, str(data).lower())
        
        if match:
            lat = float(match.group(1))
            lon = float(match.group(2))
            
            with gps_lock:
                latest_latitude = lat
                latest_longitude = lon
                
            return (lat, lon)
        
        # Alternative format: simple "lat,lon" format
        pattern = r'(-?\d+\.?\d*)[,\s]+(-?\d+\.?\d*)'
        match = re.search(pattern, str(data))
        if match:
            lat = float(match.group(1))
            lon = float(match.group(2))
            
            with gps_lock:
                latest_latitude = lat
                latest_longitude = lon
                
            return (lat, lon)
            
        return None
    except Exception as e:
        print(f"Error parsing GPS data: {e}")
        print(f"Raw data type: {type(data)}")
        print(f"Raw data preview: {str(data)[:100]}...")
        return None

def get_latest_gps_coordinates():
    """
    Get the latest GPS coordinates.
    Returns a tuple (latitude, longitude) or (None, None) if no GPS data is available.
    """
    with gps_lock:
        return (latest_latitude, latest_longitude)

def get_full_gps_data():
    """
    Get all available GPS data.
    Returns a dictionary with latitude, longitude, altitude, accuracy and timestamp.
    """
    with gps_lock:
        return {
            "latitude": latest_latitude,
            "longitude": latest_longitude,
            "altitude": latest_altitude,
            "accuracy": latest_accuracy,
            "timestamp": latest_timestamp
        }

def gps_listener():
    """
    Run the GPS listener in a separate thread.
    This function continuously listens for GPS data and updates the global variables.
    """
    global latest_latitude, latest_longitude, latest_altitude, latest_accuracy, latest_timestamp
    
    s = soc.socket()
    host = get_local_ip()
    port = 8002
    s.setsockopt(soc.SOL_SOCKET, soc.SO_REUSEADDR, 1)
    
    try:
        s.bind((host, port))
        s.listen(1)
        print(f"GPS listener started on {host}:{port}")
        
        c, addr = s.accept()
        print(f"GPS connection established with {addr}")
        
        while True:
            data = c.recv(2048).decode('utf-8')
            if not data:
                continue
                
            print(f"Received GPS data: {data}")
            
            # Parse the GPS data
            coordinates = parse_gps_data(data)
            if coordinates:
                print(f"Updated GPS coordinates: Lat {latest_latitude}, Long {latest_longitude}")
                if latest_altitude is not None:
                    print(f"Altitude: {latest_altitude}m, Accuracy: {latest_accuracy}m")
            else:
                print("Could not parse GPS data")
    
    except Exception as e:
        print(f"GPS listener error: {e}")
    finally:
        s.close()

def test_gps_connection(test_data=None):
    """
    Test GPS connection and data parsing.
    Can be used to verify that GPS data is being properly received and parsed.
    
    Args:
        test_data (str, optional): Test data to parse. If None, uses the latest received data.
        
    Returns:
        bool: True if GPS connection is working, False otherwise
    """
    if test_data:
        print(f"Testing GPS parsing with provided data: {test_data}")
        result = parse_gps_data(test_data)
        if result:
            lat, lon = result
            print(f"✓ Successfully parsed GPS test data: Lat {lat}, Long {lon}")
            return True
        else:
            print("✗ Failed to parse GPS test data")
            return False
    else:
        # Check if we have any GPS data
        lat, lon = get_latest_gps_coordinates()
        if lat is not None and lon is not None:
            print(f"✓ GPS connection is working: Latest coordinates Lat {lat}, Long {lon}")
            return True
        else:
            print("✗ No GPS data available. Check GPS connection or try sending test data.")
            return False

# GRADIO SERVER FUNCTIONALITY
def start_gradio_server():
    """Initialize and launch the Gradio interface."""
    # Get port from environment variables or use default
    port = int(os.getenv('GRADIO_PORT', 8001))
    
    iface = gr.Interface(
        fn=detect_wheat_disease,
        inputs=[
            gr.Image(type="pil", label="Upload Wheat Leaf Image"),
        ],
        outputs=gr.Image(type="pil", label="Detection Result"),
        title="Wheat Disease Detector (YOLO)",
        description="Upload a wheat leaf image to detect diseases using YOLO model.",
        article="""
        ## How to use
        1. Upload an image of a wheat leaf
        2. The model will analyze the image and detect any diseases
        3. Results with confidence above the threshold will be saved to the database

        ## About
        This application uses a YOLO model trained on wheat disease images.
        High confidence detections are automatically stored in the database via Node.js API
        and the images are uploaded to Supabase storage.
        """,
        examples=None,
        allow_flagging="never",
        analytics_enabled=False,
      
    )
    
    # Launch with compatible parameters to avoid queue-related issues
    iface.launch(
        server_port=port,
        share=False,
        debug=False,
        show_error=True
    )

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
    Also sends push notifications to registered Expo app devices.
    
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
        
        # Get GPS data from the GPS module
        gps_data = get_full_gps_data()
        latitude = gps_data['latitude']
        longitude = gps_data['longitude']
        altitude = gps_data['altitude']
        accuracy = gps_data['accuracy']
        
        print(f"GPS coordinates for detection: Lat {latitude}, Long {longitude}")
        if altitude is not None:
            print(f"Altitude: {altitude}m, Accuracy: {accuracy}m")
        
        # If GPS coordinates are not available, try to wait
        if latitude is None or longitude is None:
            print("WARNING: GPS coordinates are not available. Waiting 2 seconds for GPS data to become available...")
            time.sleep(2)
            # Refresh GPS data
            fresh_gps_data = get_full_gps_data()
            if fresh_gps_data["latitude"] is not None and fresh_gps_data["longitude"] is not None:
                # Update with fresh GPS data
                latitude = fresh_gps_data["latitude"]
                longitude = fresh_gps_data["longitude"]
                print(f"Updated GPS coordinates: Lat {latitude}, Long {longitude}")
            else:
                print("ERROR: GPS data still not available after waiting. Setting default coordinates.")
                # Set default coordinates to avoid None values
                latitude = 0.0
                longitude = 0.0
        
        # Prepare payload with image URL and direct latitude/longitude values
        payload = {
            "disease": disease_name,
            "confidence": confidence,
            "timestamp": timestamp,
            "zone_id": ZONE_ID,
            "device_id": DEVICE_ID,
            "image_url": image_url,
            "latitude": latitude,
            "longitude": longitude
        }
        
        # Store full location data for push notifications
        location_data = {
            "latitude": latitude,
            "longitude": longitude,
            "altitude": altitude,
            "accuracy": accuracy,
            "timestamp": gps_data['timestamp']
        }
        
        # Verify all required fields are present and not None
        for key, value in payload.items():
            if value is None:
                print(f"ERROR: Required field '{key}' is missing or None")
                print(f"Check your .env file to ensure related settings are properly set")
                return False
        
        print(f"Detection payload ready: {disease_name} with {confidence:.2f} confidence")
        print(f"GPS coordinates in payload: Lat {payload['latitude']}, Long {payload['longitude']}")
        
        # Send push notifications to registered devices
        if ENABLE_PUSH_NOTIFICATIONS:
            try:
                # Include location and device info in the notification
                extra_data = {
                    "timestamp": timestamp,
                    "device_id": DEVICE_ID,
                    "zone_id": ZONE_ID,
                    "location": location_data  # Use the full location data for notifications
                }
                
                # Send the push notification
                notification_result = send_disease_detection_notification(
                    disease_name,
                    confidence,
                    image_url,
                    extra_data
                )
                
                print(f"Push notification status: {notification_result['sent']} sent, {notification_result['failed']} failed")
            except Exception as e:
                print(f"Error sending push notification: {str(e)}")
        
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
            else:
                print("WARNING: No API key found. Authentication will fail!")
                print("Make sure API_KEY is set in your .env file")
            
            # Prepare request URL - ensure it points to the detections endpoint
            if not NODE_API_URL:
                print("ERROR: NODE_API_URL is not set in your .env file")
                return False
                
            DETECTIONS_URL = f"{NODE_API_URL.rstrip('/')}/detections" if not NODE_API_URL.endswith('/detections') else NODE_API_URL
            
            print(f"API URL being used: {NODE_API_URL}")
            print(f"Full endpoint URL: {DETECTIONS_URL}")
            print(f"Headers being sent: {list(headers.keys())}")
            
            # Send the request with headers
            print(f"Sending request to: {DETECTIONS_URL}")
            
            # Add a timeout to the request to avoid hanging
            response = requests.post(DETECTIONS_URL, json=payload, headers=headers, timeout=30)
            
            if response.status_code == 200:
                print(f"Successfully sent detection to Node.js API: {disease_name} ({confidence:.2f})")
                print(f"GPS coordinates sent in request: Lat {payload['latitude']}, Long {payload['longitude']}")
                return True
            else:
                print(f"Failed to send detection to Node.js API. Status code: {response.status_code}")
                print(f"Response: {response.text}")
                
                # Provide more specific error information
                if response.status_code == 401:
                    print("Authentication error: API key was not accepted")
                    print("Verify that your API_KEY in .env matches the one in your Vercel environment")
                elif response.status_code == 500:
                    print("Server error: The API server encountered an internal error")
                    print("Check the Vercel logs for more information")
                    # Try to parse the error response
                    try:
                        error_data = response.json()
                        if 'message' in error_data:
                            print(f"Error message from server: {error_data['message']}")
                    except:
                        pass
                return False
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to Node.js API server: {str(e)}")
            print(f"Exception type: {type(e).__name__}")
            print("Detection not saved. Node.js API server must be running to save detections.")
            return False
            
    except Exception as e:
        print(f"Error processing detection: {str(e)}")
        import traceback
        print(traceback.format_exc())
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

# Start threads when this module is imported
if __name__ == "__main__":
    # Start the GPS listener thread
    print("Starting GPS listener thread...")
    gps_thread = threading.Thread(target=gps_listener, daemon=True)
    gps_thread.start()
    
    # Run the Gradio server directly in the main thread
    print("Starting Gradio server in main thread...")
    start_gradio_server()
    
else:
    # When imported as a module, still start the GPS thread
    print("Starting GPS listener thread...")
    gps_thread = threading.Thread(target=gps_listener, daemon=True)
    gps_thread.start()