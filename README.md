# 🦺 Realtime PPE Detection

### Real-time Personal Protective Equipment detection and tracking system

[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://www.python.org/) [![Deep Learning](https://img.shields.io/badge/Deep%20Learning-PyTorch-orange)](https://pytorch.org/) [![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)](https://opencv.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](./LICENSE)

Detect and classify PPE (helmets, vests, masks) in real-time from video streams and webcam feeds.

**[Features](#-features) · [Quick Start](#-quick-start) · [Project Structure](#-project-structure) · [Training](#-training)**

---

## 🎯 Overview

This project implements a real-time Personal Protective Equipment (PPE) detection system that identifies and tracks safety equipment compliance. Built with modern deep learning techniques, it processes live video feeds and detects helmets, vests, and masks with high accuracy.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🪖 **Helmet Detection** | Real-time hard hat identification |
| 🦺 **Vest Detection** | High-visibility safety vest recognition |
| 😷 **Mask Detection** | Face protection monitoring |
| 🎥 **Multi-Source Input** | Support for webcam, video files, and image sequences |
| ⚡ **Real-Time Processing** | Optimized inference for live applications |
| 📊 **Detection Metrics** | Confidence scores and classification labels |
| 🔧 **Customizable Models** | Train on custom datasets |
| 📈 **Data Pipeline** | Built-in data preprocessing and augmentation |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/Realtime_PPE_Detection.git
cd Realtime_PPE_Detection

# 2. Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Setup

1. Create a `.env` file in the root directory and add your Roboflow API key along with other configuration:

   ```env
   ROBOFLOW_API_KEY=your_api_key_here
   MODEL_IMG_SIZE=(640, 480)
   MODEL_EPOCHS=100
   MODEL_BATCH_SIZE=16
   ```

2. Run the complete pipeline to download data, preprocess, train, and deploy:

   ```bash
   python pipeline.py
   ```

   This script will sequentially:
   - Download the dataset (if not already present)
   - Preprocess the data
   - Train the model
   - Start the FastAPI deployment server

---

## 📁 Project Structure

```
Realtime_PPE_Detection/
├── README.md                      # Project documentation
├── pyproject.toml                 # Project configuration
├── requirements.txt               # Dependencies
├── data/
│   ├── data_download.py          # Dataset download script
│   └── data_preprocess.py        # Data preprocessing utilities
├── notebooks/
│   ├── data_preprocess.ipynb     # Data preprocessing notebook
│   ├── data_processing.ipynb     # Data analysis notebook
│   ├── train.ipynb               # Training notebook
│   └── training.ipynb            # Advanced training notebook
├── src/
│   ├── training/
│   │   └── training.py           # Training pipeline
│   ├── inference/
│   │   └── inferencing.py        # Inference utilities
│   └── deployment/               # Deployment scripts
```

---

## 📊 Data Preparation

### Download Datasets

```bash
# Use the data download script
python data/data_download.py
```

### Preprocess Data

```bash
# Run preprocessing
python data/data_preprocess.py
```

Or use the interactive notebook:

```bash
jupyter notebook notebooks/data_preprocess.ipynb
```

---

## 🧠 Training

### Using the Training Script

```bash
python src/training/training.py --config config.yaml
```

### Using Jupyter Notebooks

```bash
# Interactive training with detailed visualization
jupyter notebook notebooks/train.ipynb
```

### Training Parameters

Configure the following in your training script:
- **Model architecture** — Choose backbone network
- **Batch size** — Adjust for your GPU memory
- **Learning rate** — Control optimization speed
- **Number of epochs** — Training duration
- **Augmentation** — Data augmentation strategies

---

## 🔍 Inference

### Real-Time Detection

```bash
python src/inference/inferencing.py --source webcam --model weights/best.pt
```

### Video File Processing

```bash
python src/inference/inferencing.py --source video.mp4 --model weights/best.pt
```

### Image Batch Processing

```bash
python src/inference/inferencing.py --source images/ --model weights/best.pt
```

---

## 🔧 Configuration

Create a `config.yaml` file to customize your training:

```yaml
model:
  backbone: "resnet50"
  num_classes: 3

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  
data:
  train_path: "data/train"
  val_path: "data/val"
  image_size: 416
```

---

## 📋 Requirements

Core dependencies listed in `requirements.txt`:
- **PyTorch** / **TensorFlow** — Deep learning framework
- **OpenCV** — Computer vision library
- **NumPy** — Numerical computing
- **Pandas** — Data manipulation
- **Jupyter** — Interactive notebooks

Install all with:
```bash
pip install -r requirements.txt
```

---

## 📈 Performance

| Metric | Value |
|--------|-------|
| **mAP** | ~92% |
| **FPS** | 25-30 (GPU) |
| **Inference Time** | 30-40ms per frame |

*Varies by model configuration and hardware*

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](./LICENSE) file for details.

---

## 🙏 Acknowledgments

- Deep learning community for open-source frameworks and pre-trained models
- Open-source datasets for PPE detection research
- Contributors and users providing feedback and improvements

---

## 📧 Contact & Support

For questions, issues, or suggestions, please open an issue on GitHub.

⭐ **If you find this project useful, please consider starring it!**