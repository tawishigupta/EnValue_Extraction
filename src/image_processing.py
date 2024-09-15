# In image_processing.py

import pytesseract
from PIL import Image

def preprocess_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        # Add any additional preprocessing steps here
        return image
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return Image.new('RGB', (224, 224))

def extract_text(image):
    return pytesseract.image_to_string(image)