from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import uvicorn
import os
from model import segment_image

app = FastAPI()

# Create a temporary folder for saving uploaded and output images if it doesn't exist
if not os.path.exists("temp"):
    os.makedirs("temp")

@app.post("/segment/")
async def segment(file: UploadFile = File(...)):
    # Save the uploaded image to a temporary location
    file_location = os.path.join("temp", file.filename)
    with open(file_location, "wb") as f:
        f.write(await file.read())
    
    # Run segmentation on the saved image
    segmented_image_path = segment_image(file_location)
    
    # Optionally remove the original uploaded file
    os.remove(file_location)
    
    # Return the segmented image
    return FileResponse(segmented_image_path, media_type="image/jpeg", filename="segmented_output.jpg")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
