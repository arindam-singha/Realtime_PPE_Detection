"""
PPE Detection Data Preprocessing Module

This module provides utilities for preprocessing PPE (Personal Protective Equipment)
detection datasets in YOLO format, including dataset loading and visualization.
"""

# Standard library imports
import glob
import os
import random
from pathlib import Path

# Third-party imports
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as T2

# ============================================================================
# Configuration
# ============================================================================

# Dataset paths
LABEL_DIR = '../data/raw/train/labels/'
IMAGE_DIR = '../data/raw/train/images/'
TARGET_SIZE = (640, 480)

# Supported image extensions
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']


# ============================================================================
# Utility Functions
# ============================================================================

def parse_label_file(file_path):
    """
    Parse YOLO format label file.
    
    Args:
        file_path (str): Path to label file
        
    Returns:
        list: List of normalized bounding box coordinates [x_center, y_center, width, height]
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:
            x_center, y_center, width, height = map(float, parts[1:])
            data.append([x_center, y_center, width, height])
    return data


def get_image_path(label_path, img_dir=IMAGE_DIR):
    """
    Find corresponding image file for a label file.
    
    Args:
        label_path (str): Path to label file
        img_dir (str): Directory containing images
        
    Returns:
        str: Path to image file, or None if not found
    """
    image_name = os.path.splitext(os.path.basename(label_path))[0]
    for ext in IMAGE_EXTENSIONS:
        img_path = os.path.join(img_dir, image_name + ext)
        if os.path.exists(img_path):
            return img_path
    return None


def visualize_dataset_samples(label_files, num_samples=9, img_dir=IMAGE_DIR):
    """
    Visualize random samples from dataset with bounding boxes.
    
    Args:
        label_files (list): List of label file paths
        num_samples (int): Number of samples to visualize
        img_dir (str): Directory containing images
    """
    rows = int(np.sqrt(num_samples))
    cols = (num_samples + rows - 1) // rows
    sampled_labels = random.sample(label_files, min(num_samples, len(label_files)))
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    axes = axes.flatten()

    for idx, label_path in enumerate(sampled_labels):
        img_path = get_image_path(label_path, img_dir)
        if img_path is None:
            axes[idx].set_title('Image not found')
            axes[idx].axis('off')
            continue
            
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        
        labels = parse_label_file(label_path)
        for class_id, obj in enumerate(labels):
            # Convert normalized coordinates to pixel values
            x_center, y_center, width, height = obj
            x = int(x_center * w)
            y = int(y_center * h)
            bw = int(width * w)
            bh = int(height * h)
            x1 = int(x - bw / 2)
            y1 = int(y - bh / 2)
            x2 = int(x + bw / 2)
            y2 = int(y + bh / 2)
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            axes[idx].text(x1, y1-5, str(class_id), color='red', fontsize=10,
                          bbox=dict(facecolor='yellow', alpha=0.5))
        
        axes[idx].imshow(img)
        axes[idx].set_title(os.path.basename(img_path))
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()


# ============================================================================
# Dataset Class
# ============================================================================

class PPEDataset(Dataset):
    """
    PyTorch Dataset for PPE detection with YOLO format labels.
    
    Handles loading images and corresponding YOLO format labels,
    with optional augmentation and transformation to target size.
    """
    
    def __init__(self, label_files, img_dir=IMAGE_DIR, target_size=TARGET_SIZE, augment=False):
        """
        Initialize PPE dataset.
        
        Args:
            label_files (list): List of paths to label files
            img_dir (str): Directory containing images
            target_size (tuple): Target image size (height, width)
            augment (bool): Whether to apply data augmentation
        """
        self.label_files = label_files
        self.img_dir = img_dir
        self.target_size = target_size
        self.augment = augment
        
        # Set up transformations
        base_transforms = [
            T2.ToImage(),
            T2.Resize(self.target_size)
        ]
        aug_transforms = [
            T2.RandomHorizontalFlip(),
            T2.RandomVerticalFlip()
        ] if augment else []
        
        self.transforms = T2.Compose(base_transforms + aug_transforms)

    @staticmethod
    def parse_label_file(file_path):
        """Parse YOLO format label file."""
        return parse_label_file(file_path)

    def get_image_path(self, label_path):
        """Get corresponding image path for a label file."""
        return get_image_path(label_path, self.img_dir)

    def __len__(self):
        """Return dataset size."""
        return len(self.label_files)

    def __getitem__(self, idx):
        """Load and process a single sample."""
        label_path = self.label_files[idx]
        img_path = self.get_image_path(label_path)
        
        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w, _ = img.shape
        
        # Parse labels
        labels = self.parse_label_file(label_path)
        
        # Convert YOLO normalized coordinates to absolute pixel values
        boxes = []
        for x_c, y_c, w, h in labels:
            x_c *= orig_w
            y_c *= orig_h
            w *= orig_w
            h *= orig_h
            # Convert to (x_min, y_min, x_max, y_max)
            x_min = x_c - w / 2
            y_min = y_c - h / 2
            x_max = x_c + w / 2
            y_max = y_c + h / 2
            boxes.append([x_min, y_min, x_max, y_max])
        
        boxes = torch.tensor(boxes, dtype=torch.float32)
        target = {
            "boxes": boxes,
            "labels": torch.ones((len(boxes),), dtype=torch.int64)
        }
        
        # Apply transforms
        img, target = self.transforms(img, target)
        
        # Convert boxes back to YOLO normalized format
        h, w = img.shape[1:]
        yolo_boxes = []
        for box in target["boxes"]:
            x_min, y_min, x_max, y_max = box.tolist()
            x_c = (x_min + x_max) / 2 / w
            y_c = (y_min + y_max) / 2 / h
            bw = (x_max - x_min) / w
            bh = (y_max - y_min) / h
            yolo_boxes.append([x_c, y_c, bw, bh])
        
        label_tensor = torch.tensor(yolo_boxes, dtype=torch.float32)
        return img, label_tensor
    

# ============================================================================
# Demo/Visualization Code
# ============================================================================

if __name__ == "__main__":
    """Demo visualization of dataset samples."""
    # Load label files
    label_files = glob.glob(os.path.join(LABEL_DIR, '*.txt'))
    
    if label_files:
        # Visualize random samples
        visualize_dataset_samples(label_files, num_samples=9, img_dir=IMAGE_DIR)
        
        # Create and test dataset
        dataset = PPEDataset(label_files, img_dir=IMAGE_DIR, augment=False)
        ppe_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        # Visualize loaded samples
        num_samples = 5
        for i, (img_tensor, label_tensor) in enumerate(ppe_loader):
            if i >= num_samples:
                break
            
            img = img_tensor[0].permute(1, 2, 0).numpy()  # Convert to HWC, RGB
            h, w, _ = img.shape
            
            plt.figure(figsize=(8, 6))
            plt.imshow(img)
            
            for label in label_tensor[0]:
                x_center, y_center, width, height = label.tolist()
                # Convert normalized to pixel coordinates
                x = int(x_center * w)
                y = int(y_center * h)
                bw = int(width * w)
                bh = int(height * h)
                x1 = int(x - bw / 2)
                y1 = int(y - bh / 2)
                
                plt.gca().add_patch(
                    plt.Rectangle((x1, y1), bw, bh, edgecolor='red',
                                 facecolor='none', linewidth=2)
                )
                plt.text(x1, y1 - 5, 'PPE', color='yellow', fontsize=10,
                        bbox=dict(facecolor='black', alpha=0.5))
            
            plt.axis('off')
            plt.title(f'Sample {i+1}')
            plt.show()
    else:
        print(f"No label files found in {LABEL_DIR}")