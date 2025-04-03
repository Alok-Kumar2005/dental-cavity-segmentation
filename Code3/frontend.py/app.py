import streamlit as st
import requests
from PIL import Image
import io

st.title("Detectron2 Image Segmentation")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Segment Image"):
        # Convert the image file to bytes
        files = {"file": uploaded_file.getvalue()}
        # Adjust the URL if your backend is hosted elsewhere
        response = requests.post("http://localhost:8000/segment/", files=files)
        if response.status_code == 200:
            result_image = Image.open(io.BytesIO(response.content))
            st.image(result_image, caption="Segmented Image", use_column_width=True)
        else:
            st.error("Error during segmentation. Please check your backend.")
