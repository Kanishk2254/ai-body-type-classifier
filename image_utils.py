"""
Image processing utilities for the Body Type Classifier.
Handles format conversion, resizing, and preprocessing.
"""

import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from typing import Optional, Tuple


class ImageProcessor:
    """Utility class for image preprocessing and format handling."""
    
    @staticmethod
    def convert_to_rgb(image: Image.Image) -> Image.Image:
        """
        Convert any image format to RGB for consistent processing.
        
        Args:
            image: PIL Image object
            
        Returns:
            RGB PIL Image object
        """
        if image.mode == 'RGB':
            return image
        
        # Handle different image modes
        if image.mode in ('RGBA', 'LA'):
            # Create white background for transparency
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'RGBA':
                rgb_image.paste(image, mask=image.split()[-1])
            else:  # LA mode
                rgb_image.paste(image, mask=image.split()[-1])
            return rgb_image
        
        elif image.mode == 'P':
            # Convert palette mode to RGBA first, then to RGB
            rgba_image = image.convert('RGBA')
            rgb_image = Image.new('RGB', rgba_image.size, (255, 255, 255))
            rgb_image.paste(rgba_image, mask=rgba_image.split()[-1])
            return rgb_image
        
        elif image.mode in ('L', 'LA'):
            # Convert grayscale to RGB
            return image.convert('RGB')
        
        else:
            # For any other mode, try direct conversion
            try:
                return image.convert('RGB')
            except Exception as e:
                print(f"Warning: Could not convert image mode {image.mode} to RGB: {e}")
                return image
    
    @staticmethod
    def save_temp_image(image: Image.Image, format='JPEG', quality=95) -> str:
        """
        Save PIL Image to a temporary file.
        
        Args:
            image: PIL Image object
            format: Output format ('JPEG', 'PNG')
            quality: JPEG quality (1-100)
            
        Returns:
            Path to temporary file
        """
        # Ensure image is in RGB mode for JPEG
        if format.upper() == 'JPEG':
            image = ImageProcessor.convert_to_rgb(image)
            suffix = '.jpg'
        else:
            suffix = '.png'
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            if format.upper() == 'JPEG':
                image.save(tmp_file.name, format, quality=quality, optimize=True)
            else:
                image.save(tmp_file.name, format)
            return tmp_file.name
    
    @staticmethod
    def preprocess_for_pose_detection(image_path: str) -> Optional[np.ndarray]:
        """
        Preprocess image for MediaPipe pose detection.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image as numpy array (RGB format)
        """
        try:
            # Load image with OpenCV
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Optional: Resize if image is too large (for performance)
            height, width = image_rgb.shape[:2]
            max_dimension = 1024
            
            if max(height, width) > max_dimension:
                if width > height:
                    new_width = max_dimension
                    new_height = int(height * (max_dimension / width))
                else:
                    new_height = max_dimension
                    new_width = int(width * (max_dimension / height))
                
                image_rgb = cv2.resize(image_rgb, (new_width, new_height), 
                                     interpolation=cv2.INTER_AREA)
            
            return image_rgb
            
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
    
    @staticmethod
    def validate_image(image_path: str) -> bool:
        """
        Validate if image file can be processed.
        
        Args:
            image_path: Path to image file
            
        Returns:
            True if image is valid, False otherwise
        """
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                return False
            
            # Try to open with PIL
            with Image.open(image_path) as img:
                # Verify the image
                img.verify()
            
            # Try to open with OpenCV
            cv_img = cv2.imread(image_path)
            if cv_img is None:
                return False
            
            return True
            
        except Exception:
            return False
    
    @staticmethod
    def get_image_info(image_path: str) -> dict:
        """
        Get information about an image file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with image information
        """
        try:
            with Image.open(image_path) as img:
                return {
                    'format': img.format,
                    'mode': img.mode,
                    'size': img.size,
                    'width': img.width,
                    'height': img.height,
                    'has_transparency': img.mode in ('RGBA', 'LA', 'P') and 'transparency' in img.info
                }
        except Exception as e:
            return {'error': str(e)}


def process_uploaded_image(uploaded_file) -> Optional[str]:
    """
    Process uploaded file from Streamlit and return temporary file path.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        Path to processed temporary file, or None if error
    """
    try:
        # Open uploaded file
        image = Image.open(uploaded_file)
        
        # Convert to RGB
        rgb_image = ImageProcessor.convert_to_rgb(image)
        
        # Save as temporary JPEG file
        temp_path = ImageProcessor.save_temp_image(rgb_image, 'JPEG', quality=95)
        
        return temp_path
        
    except Exception as e:
        print(f"Error processing uploaded image: {e}")
        return None


if __name__ == "__main__":
    # Example usage
    processor = ImageProcessor()
    
    # Test image validation
    print("Image Processor utility loaded successfully!")
    print("Available methods:")
    print("- convert_to_rgb()")
    print("- save_temp_image()")
    print("- preprocess_for_pose_detection()")
    print("- validate_image()")
    print("- get_image_info()")
