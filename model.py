import cv2
import numpy as np
import tempfile
from PIL import Image

def whiten_teeth(img, mask, factor):
    img_float = img.astype(np.float32) / 255.0
    enhanced = img_float + mask * (factor - 1)
    enhanced = np.clip(enhanced, 0, 1)
    return (enhanced * 255).astype(np.uint8)

def generate_images(uploaded_file):
    # Read uploaded file as OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Dummy mask (you should replace this with real teeth detection)
    mask = np.zeros_like(img_rgb[:, :, 0], dtype=np.float32)
    h, w = mask.shape
    # Example mask: central ellipse (simulate teeth region)
    cv2.ellipse(mask, (w//2, h//2), (w//4, h//8), 0, 0, 360, 1, -1)
    mask = np.expand_dims(mask, axis=-1)

    # Whitening levels
    factors = np.linspace(0.7, 1.5, 9)

    output_paths = []
    for i, factor in enumerate(factors):
        whitened = whiten_teeth(img_rgb, mask, factor)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        Image.fromarray(whitened).save(temp_file.name)
        output_paths.append(temp_file.name)

    return output_paths
