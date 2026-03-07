import os
from roboflow import Roboflow
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

API_KEY = os.getenv("ROBOFLOW_API_KEY")

rf = Roboflow(api_key=API_KEY)
workspace = rf.workspace()
print("Available workspace:", workspace)
project = rf.workspace("arindamsinghacomputervisionworks").project("ppe-helmet-vest-detection-vou1p-muanm")
print(project)
dataset = project.version(1).download("yolov8", location="data/raw")
print("Dataset downloaded at:", dataset.location)