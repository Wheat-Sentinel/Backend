"""
Wheat Disease Detection - Gradio Interface

This is the main entry point for the wheat disease detection application.
It provides a user-friendly interface using Gradio to upload and analyze wheat images.
"""
import gradio as gr
import os
from dotenv import load_dotenv
from detection import detect_wheat_disease

# Load environment variables from .env file
load_dotenv()

def main():
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
        allow_flagging="never"
    )
    
    iface.launch(server_port=port)

if __name__ == "__main__":
    main()