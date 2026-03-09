# utils.py
import os
from PIL import Image

def create_directory(path):
    """
    Create a directory if it does not exist.
    """
    os.makedirs(path, exist_ok=True)

def load_image(image_path):
    """
    Load an image as PIL.Image from file path.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def save_image(image, path):
    """
    Save PIL.Image to disk.
    """
    try:
        create_directory(os.path.dirname(path))
        image.save(path)
        print(f"Saved image to {path}")
    except Exception as e:
        print(f"Error saving image {path}: {e}")