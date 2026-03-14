import streamlit as st
from ultralytics import YOLO
from pathlib import Path
import shutil
import time
import os
import tempfile


# --- CONFIG ---
MODEL_DIR = Path("notebook")
MODEL_PATTERN = "*.pt"
UPLOAD_DIR = Path("uploads")
# Use absolute path for results
RESULTS_DIR = Path(r"C:/Arindam_work_new/New_folder/realtime_ppe_detection/results/predict")

UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_all_models():
    return sorted(MODEL_DIR.glob(MODEL_PATTERN), key=os.path.getmtime, reverse=True)

def save_uploaded_file(uploaded_file, save_dir):
    timestamp = int(time.time() * 1000)
    file_ext = Path(uploaded_file.name).suffix
    orig_file_stem = Path(uploaded_file.name).stem
    file_stem = orig_file_stem + f"_{timestamp}"
    file_name = f"{file_stem}{file_ext}"
    file_path = save_dir / file_name
    with open(file_path, "wb") as f:
        shutil.copyfileobj(uploaded_file, f)
    return file_path, file_stem, orig_file_stem

import shutil
def clear_results_dir():
    # Remove all subfolders and files in RESULTS_DIR
    if RESULTS_DIR.exists():
        for item in RESULTS_DIR.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

def run_inference(model_path, image_path, file_stem):
    # Clear all previous results before inference
    clear_results_dir()
    output_dir = RESULTS_DIR / file_stem
    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))
    results = model.predict(
        source=str(image_path),
        save=True,
        project=str(RESULTS_DIR),
        name=file_stem,
        exist_ok=True
    )
    # Find the predicted image in the output_dir
    predicted_images = list(output_dir.glob('*.jpg')) + list(output_dir.glob('*.png'))
    return predicted_images, output_dir

# --- STREAMLIT UI ---
st.set_page_config(page_title="PPE Detection", layout="centered")
st.title("🦺 Real-time PPE Detection")
st.markdown("""
Upload an image and select or upload a YOLOv8 model to detect PPE in the image. Results will be shown below.
""")

tab1, tab2 = st.tabs(["Inference", "Manage Models"])

with tab2:
    st.subheader("Upload Custom YOLOv8 Model")
    uploaded_model = st.file_uploader("Upload a YOLOv8 .pt model file", type=["pt"], key="model_upload")
    if uploaded_model is not None:
        model_save_path = MODEL_DIR / uploaded_model.name
        with open(model_save_path, "wb") as f:
            shutil.copyfileobj(uploaded_model, f)
        st.success(f"Model '{uploaded_model.name}' uploaded successfully!")

with tab1:
    models = get_all_models()
    model_names = [m.name for m in models]
    with st.form("inference_form"):
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="img_upload")
        model_option = st.selectbox("Select a model", model_names if model_names else ["No models found"])
        submit = st.form_submit_button("Run Inference")

    if submit:
        if not uploaded_image:
            st.error("Please upload an image.")
        elif not model_names or model_option == "No models found":
            st.error("No model available. Please add or upload a YOLOv8 model.")
        else:
            with st.spinner("Running inference..."):
                file_path, file_stem, orig_file_stem = save_uploaded_file(uploaded_image, UPLOAD_DIR)
                model_path = MODEL_DIR / model_option
                predicted_images, output_dir = run_inference(model_path, file_path, file_stem)
                matched_img_path = None
                # Show all predicted images (usually only one)
                if predicted_images:
                    for img in predicted_images:
                        st.image(str(img), caption=f"Detection Result: {img.name}")
                        with open(img, "rb") as img_file:
                            st.download_button(f"Download {img.name}", img_file, file_name=img.name)
                        # Mark the first as matched for backward compatibility
                        if matched_img_path is None:
                            matched_img_path = img
                    st.success("Inference complete!")
                else:
                    st.error("No result image found after inference.")
