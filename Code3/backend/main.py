import io
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from segmentation import segment_image

app = FastAPI()

@app.post("/segment/")
async def segment(file: UploadFile = File(...)):
    contents = await file.read()

    
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    result_img = segment_image(image)
    
    retval, buffer = cv2.imencode('.jpg', result_img)
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
