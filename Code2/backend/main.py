from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from ultralytics import YOLO
import uvicorn
import os
import shutil

app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

YOLO_MODEL = 'detection_best_point.pt'
SAM_MODEL = 'sam_b.pt'

yolo = None
sam = None

def load_models():
    global yolo, sam
    try:
        yolo = YOLO(YOLO_MODEL)
        print("YOLO Loaedddddddddddddddddd")
        sam = YOLO(SAM_MODEL) 
        print("SAM Loadedddddddddddddddddd")
    except Exception as e:
        print(f"model not loadedddddddddddd: {e}")
        raise

@app.post("/process")
async def process_image(file: UploadFile = File(...)):
    try:
        temp_path = "temp.jpg"
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        yolo_results = yolo.predict(temp_path)
        if len(yolo_results[0].boxes) == 0:
            return {"message": "No detections found"}
        
        bbox = yolo_results[0].boxes.xyxy[0].cpu().numpy()   ### getting bounding boxes
        sam_results = sam.predict(temp_path, bboxes=[bbox], save=True)    ### perdicting segmentation using box
    
        output_dir = sam_results[0].save_dir
        result_path = os.path.join(output_dir, os.listdir(output_dir)[-1])
        
        # Copy segmented image
        final_path = "segment_result.jpg"
        shutil.copy(result_path, final_path)
        
        return FileResponse(final_path)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)