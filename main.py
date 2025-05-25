# app.py
import streamlit as st
import cv2
import numpy as np
from detect_teeth import get_teeth_mask, whiten_teeth, hex_to_rgb
from PIL import Image
import io

st.set_page_config(page_title="AI Teeth Whitening Tool", layout="wide")
st.title("🦷 AI Teeth Whitening App")
st.write("Upload a photo and let the AI whiten the teeth in it.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Display original image
    st.subheader("Original Image")
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), channels="RGB", width=300)

    with st.spinner("Detecting teeth and applying whitening..."):
        # Get teeth mask
        mask = get_teeth_mask(image)
        
        # Debug: Display the mask
        st.image(mask, caption="Teeth Mask (Debug)", width=300)
    
        # Define teeth whitening colors
        # Define teeth whitening colors - don't start with black!
        whitening_colors = ['#FAF9F6', '#F4F1EC', '#EFECE6', '#EAE6DE'] 
        whitened_images = []
        
        # Generate whitened images for each color
        for color in whitening_colors:
            whitened = whiten_teeth(image, mask, color)
            whitened_images.append(whitened)

    # Display grid of whitened images
    st.subheader("Whitening Options")
    
    # Create 2x2 grid using columns and rows
    col1, col2 = st.columns(2)
    
    # First row
    with col1:
        st.image(cv2.cvtColor(whitened_images[0], cv2.COLOR_BGR2RGB), 
                channels="RGB", 
                caption=f"Level 1: {whitening_colors[0]}",
                width=300)
     
        
    with col2:
        st.image(cv2.cvtColor(whitened_images[1], cv2.COLOR_BGR2RGB), 
                channels="RGB", 
                caption=f"Level 2: {whitening_colors[1]}",
                width=300
                )
                
     
    
    # Second row
    col3, col4 = st.columns(2)
    
    with col3:
        st.image(cv2.cvtColor(whitened_images[2], cv2.COLOR_BGR2RGB), 
                channels="RGB", 
                caption=f"Level 3: {whitening_colors[2]}",
                width=300)
   
        
    with col4:
        st.image(cv2.cvtColor(whitened_images[3], cv2.COLOR_BGR2RGB), 
                channels="RGB", 
                caption=f"Level 4: {whitening_colors[3]}",
                width=300)
  