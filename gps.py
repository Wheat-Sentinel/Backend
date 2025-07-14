import socket as soc
import threading
import time
import re
import json

# Global variables to store the latest GPS coordinates
latest_latitude = None
latest_longitude = None
latest_altitude = None
latest_accuracy = None
latest_timestamp = None
gps_lock = threading.Lock()

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

# Start the GPS listener thread when the module is imported
gps_thread = threading.Thread(target=gps_listener, daemon=True)
gps_thread.start()

# If this module is run directly, just run the GPS listener in the main thread
if __name__ == "__main__":
    try:
        gps_listener()
    except KeyboardInterrupt:
        print("GPS listener stopped")