import os
import logging
from data.download import download_dataset
from data.data_processing import process_data
from src.training.train_yolov8 import train_model
import subprocess
import sys

def main():
    logging.basicConfig(level=logging.INFO)
    # Step 1: Check if data/raw exists and is not empty
    raw_data_path = "data/raw"
    if os.path.exists(raw_data_path) and os.listdir(raw_data_path):
        logging.info("Data already exists in data/raw. Skipping download.")
    else:
        logging.info("Downloading dataset...")
        download_dataset()

    # Step 2: Run data preprocessing
    logging.info("Running data preprocessing...")
    process_data()


    # Step 3: Ask user if they want to train the model
    while True:
        user_input = input("Do you want to train the model now? (y/n): ").strip().lower()
        if user_input in ['y', 'n']:
            break
        print("Please enter 'y' or 'n'.")
    if user_input == 'y':
        logging.info("Running model training...")
        train_model()
    else:
        logging.info("Skipping model training. Please ensure a trained model is available before deployment.")

    # Step 4: Run deployment (Streamlit app)
    logging.info("Starting Streamlit deployment...")

    # --- Kill any process using port 8300 (Windows only) ---
    port = 8300
    try:
        # Find process using the port
        result = subprocess.run(f'netstat -ano | findstr :{port}', shell=True, capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        pids = set()
        for line in lines:
            parts = line.split()
            if len(parts) >= 5:
                pid = parts[-1]
                if pid.isdigit():
                    pids.add(pid)
        for pid in pids:
            logging.info(f"Killing process on port {port} with PID {pid}")
            subprocess.run(f'taskkill /PID {pid} /F', shell=True)
    except Exception as e:
        logging.warning(f"Could not check/kill process on port {port}: {e}")

    # Find streamlit executable in the current virtual environment
    venv_path = os.environ.get('VIRTUAL_ENV')
    if venv_path:
        streamlit_path = os.path.join(venv_path, 'Scripts', 'streamlit.exe')
    else:
        streamlit_path = 'streamlit'  # fallback, may fail if not in PATH
    # Run Streamlit app
    process = subprocess.Popen([
        streamlit_path,
        "run",
        "src/deployment/streamlit_app.py",
        "--server.address=127.0.0.1",
        f"--server.port={port}"
    ])
    logging.info("Streamlit app started. Pipeline completed.")

if __name__ == "__main__":
    main()
    