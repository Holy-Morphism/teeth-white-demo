import streamlit as st
from model import generate_images
import os

st.title("Teeh Whitening Demo")

uploaded_file = st.file_uploader("Please upload an image of some teeth",type=["png"])



if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    st.image(bytes_data)

    images = generate_images(bytes_data)

    if images:

        top_left, top_right = st.columns(2, border=True)
        bottom_left, bottom_right = st.columns(2, border=True)


        top_left.image(images[0])
        top_right.image(images[1])
        bottom_left.image(images[2])
        bottom_right.image(images[3])

    if st.button("Reset"):
        uploaded_file = None
        for image_path in images:
            os.remove(image_path)