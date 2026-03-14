import logging
from ultralytics import YOLO

from dotenv import load_dotenv
import os

# MODEL_IMG_SIZE = (640, 480)  # Default image size
MODEL_IMG_SIZE=eval(os.getenv("MODEL_IMG_SIZE", "(640, 480)"))  # Override with env variable if set
epochs = int(os.getenv("MODEL_EPOCHS", 20))
batch_size = int(os.getenv("MODEL_BATCH_SIZE", 16))

def train_model():
    logging.info("Starting YOLOv8 training...")
    model = YOLO("notebook/yolov8s.pt")
    model.train(
        data="data/raw/data.yaml",  # Update path if needed
        imgsz=MODEL_IMG_SIZE,  # (width, height)
        epochs=epochs,
        batch=batch_size,
        device=int(os.getenv("MODEL_DEVICE", 0)),
        project="runs/detect/runs/train",
        name="yolov8_ppe"
    )
    logging.info("Training completed.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_model()
