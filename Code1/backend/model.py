from ultralytics import YOLO
import cv2
import os

# Load your fine-tuned YOLO segmentation model
model = YOLO("segment_200_epoch.pt")

def segment_image(image_path: str) -> str:
    """
    Performs segmentation on the given image and returns the path to the segmented image.
    """
    # Run inference on the image. The model returns a list of results; we take the first one.
    results = model(image_path)[0]
    # Render the segmentation results (this draws masks on the image)
    rendered_image = results.plot()  # returns an image (numpy array)
    
    # Ensure the temporary folder exists
    output_dir = "temp"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the rendered (segmented) image
    output_path = os.path.join(output_dir, "segmented_output.jpg")
    cv2.imwrite(output_path, rendered_image)
    return output_path
