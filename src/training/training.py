from ultralytics import YOLO

model = YOLO("yolov8s.pt")

model.train(
    data="../data/raw/data.yaml",  # Update path if needed
    imgsz=(640, 480),  # (width, height)
    epochs=2,
    batch=16,
    project="runs/train",
    name="yolov8_ppe"
)

