from pathlib import Path
import shutil
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from src.inference.predict import predict_image


app = FastAPI(
    title="Traffic Object Detection API",
    version="1.0.0"
)


UPLOAD_DIR = Path("runs/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

PREDICT_DIR = Path("runs/predict")
PREDICT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@app.get("/")
def root():
    return {
        "message": "Traffic Object Detection API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.get("/prediction/{filename}")
def get_prediction_image(filename: str):
    file_path = PREDICT_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(file_path)


@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    file_extension = Path(file.filename).suffix.lower()

    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only .jpg, .jpeg, .png files are allowed"
        )

    unique_filename = f"{uuid.uuid4().hex}{file_extension}"
    saved_file_path = UPLOAD_DIR / unique_filename

    with saved_file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = predict_image(
            image_path=str(saved_file_path),
            conf_threshold=0.5,
            save_annotated=True,
            output_dir=str(PREDICT_DIR)
        )

        prediction_file = Path(result["annotated_image_path"]).name if result["annotated_image_path"] else None

        return JSONResponse(
            content={
                "filename": file.filename,
                "saved_file_path": str(saved_file_path),
                "prediction_image_url": f"/prediction/{prediction_file}" if prediction_file else None,
                "detections": result["detections"],
                "detections_count": len(result["detections"])
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}"
        )