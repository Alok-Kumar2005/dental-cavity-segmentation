import streamlit as st
import requests
from PIL import Image
import io

st.title("Dental Image Segmentation")
st.write("Upload a dental image for analysis")

uploaded_file = st.file_uploader("Choose image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_column_width=True)
    
    if st.button("Process"):
        try:
            response = requests.post(
                "http://localhost:8000/process",
                files={"file": uploaded_file.getvalue()}
            )
            if response.status_code == 200:
                result_img = Image.open(io.BytesIO(response.content))
                st.image(result_img, caption="Segmentation Result", use_column_width=True)
            else:
                st.error(f"Error: {response.text}")
                
        except requests.ConnectionError:
            st.error("Backend not running! Start it first with:")
            st.code("uvicorn backend.app:app --reload")