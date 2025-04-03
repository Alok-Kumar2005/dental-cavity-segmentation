import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import os
from PIL import Image
import io

def main():
    st.title("Image Segmentation")
    st.write("Upload Image")
    ###### file uploading
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    ### loading model
    model_path = st.text_input("Model Path", value="segment_200_epoch.pt")
    
    if uploaded_file is not None:
        ###### converting input image to file
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Display the original image
        st.subheader("Original Image")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption='Original Image', use_column_width=True)
        
        if st.button("Process Image"):
            with st.spinner("Processing..."):
                try:
                    model = YOLO(model_path)
                    H, W, _ = img.shape
                    results = model.predict(source=img)
                    mask_found = False
                    
                    for i, result in enumerate(results):
                        if hasattr(result, 'masks') and result.masks is not None:
                            mask_found = True
                            for j, mask in enumerate(result.masks.data):
                                mask_np = mask.cpu().numpy() * 255  ### converting mask to numpy 
                                mask_resized = cv2.resize(mask_np, (W, H)).astype(np.uint8)
                                
                                # Create a 3-channel mask for applying to the color image
                                # mask_3channel = cv2.merge([mask_resized, mask_resized, mask_resized]) / 255.0
                                


                                ##### applying mask to img and mask_resized to get the actaul cavity
                                segmented_img = cv2.bitwise_and(img, img, mask=mask_resized)
                                

                                color_overlay = np.zeros_like(img)
                                color_overlay[:,:,1] = mask_resized  
                                
                                
                                #### blending img and overlay image
                                alpha = 0.4
                                highlighted_img = cv2.addWeighted(img, 1, color_overlay, alpha, 0)
                                

                                ## outputs
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.subheader(f"Mask {j}")
                                    st.image(mask_resized, caption=f'Binary Mask {j}', use_column_width=True)
                                
                                with col2:
                                    st.subheader(f"Segmented {j}")
                                    st.image(cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB), 
                                            caption=f'Segmented Image {j}', 
                                            use_column_width=True)
                                
                                with col3:
                                    st.subheader(f"Highlighted {j}")
                                    st.image(cv2.cvtColor(highlighted_img, cv2.COLOR_BGR2RGB), 
                                            caption=f'Highlighted Image {j}', 
                                            use_column_width=True)
                                
                                st.subheader("Download Images")
                                
                                # Convert images into byytes
                                mask_bytes = cv2.imencode('.png', mask_resized)[1].tobytes()
                                segmented_bytes = cv2.imencode('.png', segmented_img)[1].tobytes()
                                highlighted_bytes = cv2.imencode('.png', highlighted_img)[1].tobytes()
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.download_button(
                                        label=f"Download Mask {j}",
                                        data=mask_bytes,
                                        file_name=f'mask_{j}.png',
                                        mime="image/png"
                                    )
                                
                                with col2:
                                    st.download_button(
                                        label=f"Download Segmented {j}",
                                        data=segmented_bytes,
                                        file_name=f'segmented_{j}.png',
                                        mime="image/png"
                                    )
                                
                                with col3:
                                    st.download_button(
                                        label=f"Download Highlighted {j}",
                                        data=highlighted_bytes,
                                        file_name=f'highlighted_{j}.png',
                                        mime="image/png"
                                    )
                    
                    if not mask_found:
                        st.error("NO maks found")
                
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()