import streamlit as st
from ultralytics import YOLO, SAM
from PIL import Image
import tempfile
import os
import shutil

st.title("Dental Image Segmentation")
st.write("Upload a dental image for analysis")

@st.cache_resource
def load_models():
    try:
        yolo = YOLO('detection_best_point.pt')
        sam = SAM('sam_b.pt')
        st.success("Models loaded successfully!")
        return yolo, sam
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        raise

yolo_model, sam_model = load_models()

uploaded_file = st.file_uploader("Choose image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_column_width=True)
    
    if st.button("Process"):
        with st.spinner("Processing..."):
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    image.save(tmp, format="JPEG")
                    temp_path = tmp.name
                yolo_results = yolo_model(temp_path)
                
                if len(yolo_results[0].boxes) == 0:
                    st.warning("No dental features detected")
                else:
                    boxes = yolo_results[0].boxes.xyxy
                    sam_result = sam_model(
                        yolo_results[0].orig_img,
                        bboxes=boxes,
                        save=True,
                        verbose=False,
                        device='cpu'  
                    )
                    
                    output_dir = sam_result[0].save_dir
                    result_path = os.path.join(output_dir, os.listdir(output_dir)[-1])
                    result_img = Image.open(result_path)
                    st.image(result_img, caption="Segmentation Result", use_column_width=True)
                    
                    shutil.rmtree(output_dir)
                
            except Exception as e:
                st.error(f"Processing failed: {str(e)}")
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.unlink(temp_path)


