from fastapi import APIRouter, UploadFile, File
from PIL import Image
import io
from app.preprocessing import preprocess_image
from app.inference import predict

router = APIRouter(
    prefix="/predict",  
    tags=["Posts"]
)

@router.post("/")
async def predict_flower(file: UploadFile = File(...)):
    try:
        # Read and process the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        processed = preprocess_image(image)
        
        # Get prediction from your model
        result = predict(processed)
        
        # Return the prediction in a consistent format
        return result
    except Exception as e:
        return {"error": str(e)}
