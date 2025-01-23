import torch
from diffusers import StableDiffusionPipeline
import os
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import time

class ImageGenerator:
    def __init__(self):
        # Initialize the model
        self.model_id = "runwayml/stable-diffusion-v1-5"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = self.load_model()
        
    def load_model(self):
        """Load the Stable Diffusion model"""
        print("Loading Stable Diffusion model...")
        pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        pipe = pipe.to(self.device)
        # Enable memory optimization
        pipe.enable_attention_slicing()
        return pipe
    
    def generate_image(self, prompt, output_path="generated_images"):
        """Generate an image from a text prompt"""
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_path, exist_ok=True)
            
            print(f"Generating image for prompt: {prompt}")
            # Generate the image
            image = self.pipe(
                prompt,
                num_inference_steps=30,  # Lower for faster generation, increase for better quality
                guidance_scale=7.5
            ).images[0]
            
            # Save the image
            filename = f"{output_path}/generated_image.png"
            image.save(filename)
            print(f"Image saved successfully at: {filename}")
            return filename
            
        except Exception as e:
            print(f"Error generating image: {str(e)}")
            return None

def start_server(port=8000):
    """Start a simple HTTP server"""
    # Change to the directory containing the files
    current_dir = os.getcwd()
    
    class CustomHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=current_dir, **kwargs)
    
    try:
        server = HTTPServer(('127.0.0.1', port), CustomHandler)
        print(f"Server started at http://127.0.0.1:{port}")
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.daemon = True
        server_thread.start()
        return server
    except Exception as e:
        print(f"Error starting server: {e}")
        return None

def create_html_viewer(image_path):
    """Create an HTML file to display the image"""
    # Convert to relative path
    relative_path = os.path.relpath(image_path)
    relative_path = relative_path.replace('\\', '/')  # Fix for Windows paths
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Generated Image</title>
        <style>
            body {{
                display: flex;
                flex-direction: column;
                align-items: center;
                background-color: #f0f0f0;
                font-family: Arial, sans-serif;
            }}
            img {{
                max-width: 800px;
                margin: 20px;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0,0,0,0.2);
            }}
            .buttons {{
                margin: 20px;
            }}
            button {{
                padding: 10px 20px;
                margin: 0 10px;
                cursor: pointer;
            }}
        </style>
    </head>
    <body>
        <h1>Generated Image</h1>
        <img src="{relative_path}" alt="Generated Image">
        <div class="buttons">
            <button onclick="window.open('{relative_path}', '_blank')">Open in New Tab</button>
        </div>
    </body>
    </html>
    """
    
    html_path = "view_image.html"
    with open(html_path, "w") as f:
        f.write(html_content)
    return html_path

def open_in_browsers(html_path):
    """Open the HTML file in multiple browsers"""
    # Use 127.0.0.1 instead of localhost
    url = f"http://127.0.0.1:8000/{html_path}"
    
    print(f"Opening URL: {url}")
    
    # Wait a bit longer for server to start
    time.sleep(2)
    
    # Open in default browser (usually Chrome)
    webbrowser.open(url)
    
    # Open in Edge (Windows)
    try:
        edge_path = "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
        if os.path.exists(edge_path):
            webbrowser.get(f'"{edge_path}" %s').open(url)
    except Exception as e:
        print(f"Could not open Edge: {e}")

def main():
    # Create generator instance
    generator = ImageGenerator()
    
    # Get prompt from user
    prompt = input("Enter your image description: ")
    
    # Generate image
    generated_image = generator.generate_image(prompt)
    
    if generated_image:
        print("\nImage generation completed successfully!")
        
        # Start local server
        server = start_server()
        
        if server:
            # Create HTML viewer
            html_path = create_html_viewer(generated_image)
            
            # Open in browsers
            open_in_browsers(html_path)
            
            # Keep the script running
            input("\nPress Enter to exit...")
            server.shutdown()
        else:
            print("Failed to start server. Please check if port 8000 is available.")

if __name__ == "__main__":
    main()
