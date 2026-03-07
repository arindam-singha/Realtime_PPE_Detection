import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from typing import List, Optional
import shutil
import glob
from ultralytics import YOLO
import uuid
from pathlib import Path


# --- CONFIG ---
MODEL_DIR = Path("notebook")
MODEL_PATTERN = "yolov8*.pt"
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("C:/Arindam_work_new/New_folder/realtime_ppe_detection/results")

UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

app = FastAPI()
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")
templates = Jinja2Templates(directory="src/deployment/templates")

# --- UTILS ---
def get_latest_model():
    models = sorted(MODEL_DIR.glob(MODEL_PATTERN), key=os.path.getmtime, reverse=True)
    return models[0] if models else None

def get_all_models():
    return sorted(MODEL_DIR.glob(MODEL_PATTERN), key=os.path.getmtime, reverse=True)

# --- ROUTES ---

# Home page: select model (from list or upload), upload image
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    models = get_all_models()
    return templates.TemplateResponse("index.html", {"request": request, "models": models})


# Inference endpoint: select model from dropdown or upload model file
@app.post("/infer", response_class=HTMLResponse)
async def infer(
    request: Request,
    file: UploadFile = File(...),
    model_name: Optional[str] = Form(None),
    model_file: Optional[UploadFile] = File(None)
):
    # Save uploaded image
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Determine model path
    if model_file is not None and model_file.filename:
        # User uploaded a model file
        uploaded_model_path = UPLOAD_DIR / model_file.filename
        with open(uploaded_model_path, "wb") as buffer:
            shutil.copyfileobj(model_file.file, buffer)
        model_path = uploaded_model_path
    elif model_name:
        model_path = MODEL_DIR / model_name
    else:
        return templates.TemplateResponse("index.html", {"request": request, "models": get_all_models(), "error": "Please select or upload a model file."})

    # Use Ultralytics YOLOv8 API for inference and result saving
    try:
        model = YOLO(str(model_path))
        # Use a unique subfolder for each inference to avoid collisions
        unique_id = str(uuid.uuid4())
        output_dir = RESULTS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        # Run prediction and force save to output_dir
        results = model.predict(
            source=str(file_path),
            save=True,
            project=str(output_dir),
            name="",
            exist_ok=True
        )
        # Debug: print output_dir contents
        print(f"[DEBUG] Output dir: {output_dir}")
        print(f"[DEBUG] Files: {os.listdir(output_dir)}")
        # Find the result image in the output_dir or its subfolders
        result_images = list(output_dir.glob('**/*.jpg')) + list(output_dir.glob('**/*.png'))
        if result_images:
            # Copy the result image to RESULTS_DIR for static serving
            result_img_path = RESULTS_DIR / ('_' + result_images[0].name)
            shutil.copy(result_images[0], result_img_path)
            print(f"[DEBUG] Result image copied to: {result_img_path}")
            return templates.TemplateResponse("result.html", {
                "request": request,
                "result_img": f"/results/{result_img_path.name}",
                "download_img": f"/results/{result_img_path.name}"
            })
        else:
            print(f"[DEBUG] No result images found in {output_dir}")
            return templates.TemplateResponse("result.html", {"request": request, "result_img": None, "download_img": None})
    except Exception as e:
        print(f"[ERROR] Model loading or inference failed: {e}")
        return templates.TemplateResponse("index.html", {"request": request, "models": get_all_models(), "error": f"Model loading or inference failed: {e}"})

# --- TEMPLATES ---
# Place index.html and result.html in src/deployment/templates/
# index.html: Form to upload image and select model
# result.html: Show result image
