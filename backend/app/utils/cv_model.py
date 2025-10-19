import numpy as np
from PIL import Image
import io
import requests

def extract_image_features(image_url):
    """
    Extract simple color histogram features from an image URL.
    Returns a normalized feature vector (1D numpy array).
    """
    try:
        response = requests.get(image_url, timeout=5)
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        img = img.resize((128, 128))
        hist = np.array(img.histogram(), dtype=np.float32)
        hist /= hist.sum()  # normalize
        return hist
    except Exception as e:
        print("Image feature extraction failed:", e)
        return np.zeros(256 * 3, dtype=np.float32)
