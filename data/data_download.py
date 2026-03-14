import os
import logging
from roboflow import Roboflow
from dotenv import load_dotenv

def download_dataset():
	load_dotenv()  # Load environment variables from .env file
	API_KEY = os.getenv("ROBOFLOW_API_KEY")
	if not API_KEY:
		logging.error("ROBOFLOW_API_KEY not found in environment variables.")
		return
	rf = Roboflow(api_key=API_KEY)
	workspace = rf.workspace()
	logging.info(f"Available workspace: {workspace}")
	project = rf.workspace("arindamsinghacomputervisionworks").project("ppe-helmet-vest-detection-vou1p-muanm")
	logging.info(f"Project: {project}")
	dataset = project.version(1).download("yolov8", location="data/raw")
	logging.info(f"Dataset downloaded at: {dataset.location}")

if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO)
	download_dataset()