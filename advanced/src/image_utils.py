# src/image_utils.py
import requests
import os
from dotenv import load_dotenv
from diffusers import StableDiffusionPipeline
import torch

load_dotenv()

class ImageRetriever:
    def __init__(self):
        self.unsplash_access_key = os.getenv("UNSPLASH_ACCESS_KEY")
        self.pixabay_api_key = os.getenv("PIXABAY_API_KEY")
        self.unsplash_base_url = "https://api.unsplash.com/search/photos"
        self.pixabay_base_url = "https://pixabay.com/api/"
        self.stable_diffusion_model = None
    
    def get_relevant_image(self, theme, count=1):
        """
        Retrieve relevant images from Unsplash based on theme
        
        Args:
            theme (str): Topic/theme to search images for
            count (int): Number of images to retrieve
        
        Returns:
            list: URLs of retrieved images
        """
        if not self.unsplash_access_key:
            print("Warning: No Unsplash API key configured")
            return []
        
        # Sanitize theme
        clean_theme = ''.join(
            char for char in theme.lower().replace('\n', ' ').strip() 
            if char.isalnum() or char.isspace()
        )[:50]
        
        # Fallback to generic theme if sanitization fails
        clean_theme = clean_theme or "business"
        
        params = {
            "query": clean_theme,
            "client_id": self.unsplash_access_key,
            "per_page": count,
            "orientation": "squarish"
        }
        
        try:
            response = requests.get(self.unsplash_base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            images = [
                {
                    "url": photo["urls"]["regular"],
                    "description": photo.get("description", clean_theme),
                    "alt_description": photo.get("alt_description", clean_theme)
                } 
                for photo in data.get("results", [])
            ]
            
            return images
        except requests.RequestException as e:
            print(f"Unsplash image retrieval error: {e}")
            return []
        
    def get_pixabay_images(self, theme, count=1):
        """
        Retrieve relevant images from Pixabay based on theme
        """
        # Ensure valid API parameters
        if not self.pixabay_api_key:
            print("Warning: No Pixabay API key configured")
            return []
        
        # Sanitize theme: remove newlines, limit length, remove special characters
        clean_theme = ''.join(
            char for char in theme.lower().replace('\n', ' ').strip() 
            if char.isalnum() or char.isspace()
        )[:50]  # Limit to 50 characters
        
        # Fallback to generic theme if sanitization fails
        clean_theme = clean_theme or "business"
        
        params = {
            "key": self.pixabay_api_key,
            "q": clean_theme,
            "image_type": "photo",
            "safesearch": "true"
        }
        
        try:
            response = requests.get(self.pixabay_base_url, params=params, timeout=10)
            
            # Debug print
            print(f"Pixabay API Request URL: {response.url}")
            print(f"Pixabay API Response Status: {response.status_code}")
            print(f"Pixabay API Response Content: {response.text}")
            
            response.raise_for_status()
            
            data = response.json()
            
            # More robust image extraction
            images = []
            for hit in data.get("hits", [])[:count]:
                if hit.get("webformatURL"):
                    images.append({
                        "url": hit["webformatURL"],
                        "description": hit.get("tags", clean_theme),
                        "alt_description": hit.get("tags", clean_theme)
                    })
            
            # If no images found, log a warning
            if not images:
                print(f"No images found for theme: {clean_theme}")
            
            return images
        
        except requests.RequestException as e:
            print(f"Pixabay image retrieval error: {e}")
            return []
        
    def load_stable_diffusion(self):
        """Load Stable Diffusion model with robust error handling"""
        if not self.stable_diffusion_model:
            try:
                # Load with float32 instead of float16
                self.stable_diffusion_model = StableDiffusionPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-xl-base-1.0", 
                    torch_dtype=torch.float32,  # Changed from float16
                    low_cpu_mem_usage=True,
                    device_map="auto"
                )
                
                print("Stable Diffusion model loaded successfully")
                return self.stable_diffusion_model
                
            except Exception as e:
                print(f"Comprehensive Stable Diffusion loading error: {e}")
                import traceback
                traceback.print_exc()
                
                # Fallback to a different model or method
                try:
                    self.stable_diffusion_model = StableDiffusionPipeline.from_pretrained(
                        "runwayml/stable-diffusion-v1-5",  # Alternative model
                        torch_dtype=torch.float32  # Ensure float32
                    )
                    print("Fallback model loaded successfully")
                    return self.stable_diffusion_model
                except Exception as fallback_error:
                    print(f"Fallback loading failed: {fallback_error}")
                    return None
        
        return self.stable_diffusion_model

    def generate_ai_image(self, theme, count=1):
        """Generate AI image using Stable Diffusion with comprehensive error handling"""
        print(f"Attempting to generate AI image for theme: {theme}")

        # Debug device information
        print("Torch version:", torch.__version__)
        print("Current device: CPU")

        model = self.load_stable_diffusion()
        if not model:
            print("Failed to load Stable Diffusion model")
            return []
        
        # Generate image with theme
        prompt = f"High-quality photorealistic image of {theme}, professional, detailed"
        
        try:
            # Ensure model is on CPU
            model = model.to("cpu")
            
            # Modify generation parameters
            images = model(
                prompt=prompt, 
                num_inference_steps=30,  # Reduced from 50 
                guidance_scale=7.5, 
                negative_prompt="low quality, blurry, sketch",
                height=512,  # Explicitly set dimensions
                width=512
            ).images
            
            # Save and prepare image URLs
            ai_images = []
            for i, image in enumerate(images[:count]):
                # Create a directory for AI-generated images if it doesn't exist
                os.makedirs("ai_generated_images", exist_ok=True)
                
                # Save image
                filename = f"ai_generated_images/ai_generated_{theme}_{i}.png"
                image.save(filename)
                
                ai_images.append({
                    "url": filename,
                    "description": f"AI-generated image for {theme}",
                    "alt_description": f"AI image: {theme}"
                })
            
            print(f"Successfully generated {len(ai_images)} AI images")
            return ai_images
        
        except Exception as e:
            print(f"Comprehensive AI image generation error: {e}")
            import traceback
            traceback.print_exc()
            return []