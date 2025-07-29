"""
Expo Push Notification Handler for Wheat Disease Detection

This module handles sending push notifications to the Expo React Native app
when wheat diseases are detected.
"""
import os
import pymongo
from exponent_server_sdk import (
    DeviceNotRegisteredError,
    PushClient,
    PushMessage,
    PushServerError,
    PushTicketError,
)
from requests.exceptions import ConnectionError, HTTPError
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# Get push notification configuration from environment variables
EXPO_ACCESS_TOKEN = os.getenv('EXPO_ACCESS_TOKEN')  # Optional for higher rate limits

# MongoDB configuration
MONGO_URI = os.getenv('MONGO_URI')
DATABASE_NAME = os.getenv('DATABASE_NAME', 'wheat_disease_detection')

class ExpoNotificationManager:
    """Manages sending push notifications to Expo clients"""
    
    def __init__(self):
        """Initialize the Expo notification manager"""
        self.client = PushClient(access_token=EXPO_ACCESS_TOKEN)
        self.default_sound = "default"
        
    def send_detection_notification(self, push_token, disease_name, confidence, image_url=None, extra_data=None):
        """
        Send a push notification about a disease detection to a specific device
        
        Args:
            push_token (str): The Expo push token of the recipient device
            disease_name (str): Name of the detected disease
            confidence (float): Confidence score of the detection (0-1)
            image_url (str, optional): URL to the detection image
            extra_data (dict, optional): Any additional data to send
            
        Returns:
            dict: The response from the Expo push service
        """
        # Format the confidence score as a percentage
        confidence_percent = int(confidence * 100)
        
        # Prepare the notification message
        title = f"Wheat Disease Alert: {disease_name}"
        body = f"Detected with {confidence_percent}% confidence. Tap to view details."
        
        # Prepare additional data to pass to the app
        data = {
            "disease": disease_name,
            "confidence": confidence,
            "imageUrl": image_url
        }
        
        # Add any extra data if provided
        if extra_data and isinstance(extra_data, dict):
            data.update(extra_data)
            
        # Send the notification
        try:
            response = self.send_push_message(
                push_token,
                title=title,
                message=body,
                data=data
            )
            print(f"Push notification sent: {title}")
            return response
        except Exception as e:
            print(f"Error sending push notification: {str(e)}")
            return {"error": str(e)}
    
    def send_push_message(self, token, title, message, data=None, sound=None, badge=None):
        """
        Send a push message to a device with the given token
        
        Args:
            token (str): The Expo push token
            title (str): Notification title
            message (str): Notification body message
            data (dict, optional): Additional data to send with the notification
            sound (str, optional): Sound to play with the notification
            badge (int, optional): App icon badge number
            
        Returns:
            dict: Response from the Expo push service
        """
        # Validate the token format (should be ExponentPushToken[xxxxxxxxxxxxxxxxxxxxxx])
        if not token or not isinstance(token, str) or not token.startswith('ExponentPushToken['):
            print(f"Invalid Expo push token format: {token}")
            return {"error": "Invalid push token format"}
            
        # Create the push message
        push_message = PushMessage(
            to=token,
            title=title,
            body=message,
            data=data or {},
            sound=sound or self.default_sound,
            badge=badge,
        )
        
        # Try to send the message
        try:
            response = self.client.publish(push_message)
            return {"success": True, "ticket": response.id}
        except PushServerError as exc:
            # Encountered some likely temporary error
            return {"success": False, "error": f"Push server error: {exc}"}
        except (ConnectionError, HTTPError) as exc:
            # Encountered some Connection or HTTP error
            return {"success": False, "error": f"Connection error: {exc}"}
        except PushTicketError as exc:
            # Encountered error with the push ticket
            return {"success": False, "error": f"Push ticket error: {exc}"}
        except DeviceNotRegisteredError:
            # The device token is no longer valid, you should remove it from your database
            return {"success": False, "error": "Device no longer registered", "unregister": True}
        except Exception as exc:
            # Encountered some other error
            return {"success": False, "error": f"Unknown error: {exc}"}

# Function to load registered tokens from MongoDB
def load_push_tokens():
    """
    Load registered push tokens from MongoDB
    
    Returns:
        list: List of Expo push tokens
    """
    try:
        # Connect to MongoDB
        client = pymongo.MongoClient(MONGO_URI)
        db = client[DATABASE_NAME]
        tokens_collection = db['push_tokens']
        
        # Query for all tokens
        token_documents = tokens_collection.find({}, {"token": 1, "_id": 0})
        
        # Extract just the token strings
        tokens = [doc['token'] for doc in token_documents]
        
        # Close the connection
        client.close()
        
        return tokens
    except Exception as e:
        print(f"Error loading push tokens from database: {str(e)}")
        return []

# Function to mark tokens as invalid in the database
def mark_token_as_invalid(token):
    """
    Mark a push token as invalid in the database
    
    Args:
        token (str): The Expo push token to mark as invalid
    
    Returns:
        bool: True if marked successfully, False otherwise
    """
    try:
        # Connect to MongoDB
        client = pymongo.MongoClient(MONGO_URI)
        db = client[DATABASE_NAME]
        tokens_collection = db['push_tokens']
        
        # Update the token status
        result = tokens_collection.update_one(
            {"token": token},
            {"$set": {"valid": False, "invalidated_at": datetime.now()}}
        )
        
        # Close the connection
        client.close()
        
        return result.modified_count > 0
    except Exception as e:
        print(f"Error marking token as invalid: {str(e)}")
        return False

# Create a singleton instance
notification_manager = ExpoNotificationManager()

# Function to send detection notifications to all registered devices
def send_disease_detection_notification(disease_name, confidence, image_url=None, extra_data=None):
    """
    Send a push notification about a disease detection to all registered devices
    
    Args:
        disease_name (str): Name of the detected disease
        confidence (float): Confidence score of the detection (0-1)
        image_url (str, optional): URL to the detection image
        extra_data (dict, optional): Any additional data to send
        
    Returns:
        dict: Summary of notification attempts
    """
    tokens = load_push_tokens()
    results = {
        "total": len(tokens),
        "sent": 0,
        "failed": 0,
        "errors": []
    }
    
    if not tokens:
        print("No push tokens registered in database. Notifications not sent.")
        results["error"] = "No push tokens registered"
        return results
        
    for token in tokens:
        try:
            response = notification_manager.send_detection_notification(
                token, disease_name, confidence, image_url, extra_data
            )
            
            if response.get("success", False):
                results["sent"] += 1
            else:
                results["failed"] += 1
                results["errors"].append({
                    "token": token,
                    "error": response.get("error", "Unknown error")
                })
                
                # If the device is no longer registered, mark it as invalid in the database
                if response.get("unregister", False):
                    mark_token_as_invalid(token)
                
        except Exception as e:
            results["failed"] += 1
            results["errors"].append({
                "token": token,
                "error": str(e)
            })
            
    print(f"Push notification summary: {results['sent']} sent, {results['failed']} failed")
    return results