import cv2
import numpy as np
import mediapipe as mp

def get_teeth_mask(image):
    """
    Create a mask for the teeth region in a mouth image
    using color and intensity-based detection
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Convert to different color spaces for better teeth detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # 1. Use color thresholding to find white/light pixels (teeth)
    # HSV range for white/light teeth
    lower_white_hsv = np.array([0, 0, 170])  # Low saturation, high value
    upper_white_hsv = np.array([180, 65, 255])
    white_mask_hsv = cv2.inRange(hsv, lower_white_hsv, upper_white_hsv)
    
    # LAB range for teeth (L: lightness, high for teeth)
    l_channel = lab[:,:,0]
    a_channel = lab[:,:,1]
    b_channel = lab[:,:,2]
    
    # Threshold on lightness channel
    _, l_thresh = cv2.threshold(l_channel, 175, 255, cv2.THRESH_BINARY)
    
    # 2. Combine masks
    combined_mask = cv2.bitwise_and(white_mask_hsv, l_thresh)
    
    # 3. Refine mask with morphological operations
    kernel = np.ones((3,3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 4. Find contours and filter by size and position
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 50:  # Filter out very small contours
            continue
            
        # Get bounding box
        x, y, w_rect, h_rect = cv2.boundingRect(contour)
        
        # Calculate center position
        center_y = y + h_rect // 2
        
        # Only keep contours in the middle portion of the image
        if (center_y > h * 0.3 and center_y < h * 0.8 and 
            x > w * 0.1 and x + w_rect < w * 0.9):
            filtered_contours.append(contour)
    
    # 5. Draw the filtered contours on the mask
    cv2.drawContours(mask, filtered_contours, -1, 255, -1)
    
    # 6. If no teeth found with the above method, use a fallback approach
    if np.sum(mask) < 100:  # Very little or no teeth detected
        # Create a simple elliptical mask in the center
        center_ellipse = (w // 2, h // 2)
        axes = (w // 3, h // 4)
        cv2.ellipse(mask, center_ellipse, axes, 0, 0, 360, 255, -1)
    
    # Save debug image
    cv2.imwrite("teeth_mask_debug.png", mask)
    return mask

def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple"""
    h = hex_color.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def whiten_teeth(image, mask, target_color='#F4F1EC'):
    """
    Apply whitening to teeth with a target color and soft edges
    
    Args:
        image: Input image (BGR format)
        mask: Teeth mask (single channel)
        target_color: Hex color for teeth in format '#RRGGBB'
    """
    # Convert hex to BGR (OpenCV uses BGR)
    target_rgb = hex_to_rgb(target_color)
    target_bgr = (target_rgb[2], target_rgb[1], target_rgb[0])
    
    # Create a feathered mask with soft edges
    feather_amount = 7  # Adjust based on image resolution
    mask_float = mask.astype(np.float32) / 255.0
    
    # Apply gaussian blur to create soft edges
    feathered_mask = cv2.GaussianBlur(mask_float, (feather_amount*2+1, feather_amount*2+1), 0)
    
    # Work in LAB color space for better color manipulation
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Create 3D mask for vector operations
    mask_3d = np.expand_dims(feathered_mask, axis=2).repeat(3, axis=2)
    
    # Increase lightness with soft transition based on mask intensity
    l_boost = 15  # Reduced boost amount for subtlety
    l_float = l.astype(np.float32)
    l_float += feathered_mask * l_boost
    l = np.clip(l_float, 0, 255).astype(np.uint8)
    
    # Reduce yellow tint (b channel in LAB) gradually based on mask intensity
    b_reduction = 8
    b_float = b.astype(np.float32)
    b_float -= feathered_mask * b_reduction
    b = np.clip(b_float, 0, 255).astype(np.uint8)
    
    # Merge channels
    whitened_lab = cv2.merge([l, a, b])
    whitened_image = cv2.cvtColor(whitened_lab, cv2.COLOR_LAB2BGR)
    
    # Create a blended image with target color
    overlay = np.zeros_like(image, dtype=np.float32)
    overlay[:] = target_bgr
    
    # Apply color blend with varying alpha based on mask intensity
    alpha = 0.25  # Reduced for subtlety
    
    # Convert to float for better blending
    img_float = image.astype(np.float32)
    whitened_float = whitened_image.astype(np.float32)
    overlay_float = overlay.astype(np.float32)
    
    # Weighted blending with mask intensity
    blend = whitened_float * (1 - alpha * mask_3d) + overlay_float * (alpha * mask_3d)
    
    # Final result with gradual transition
    result_float = img_float * (1 - mask_3d) + blend * mask_3d
    result = np.clip(result_float, 0, 255).astype(np.uint8)
    
    # Add a hint of original texture back for realism
    texture_alpha = 0.3
    texture_mask = feathered_mask * texture_alpha
    texture_mask_3d = np.expand_dims(texture_mask, axis=2).repeat(3, axis=2)
    
    result = result * (1 - texture_mask_3d) + image.astype(np.float32) * texture_mask_3d
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result
   