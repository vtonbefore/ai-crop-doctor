import json
import os

class Config:
    DATA_DIR = "./data/train"    # Your training data folder, organized by class subfolders
    MODEL_DIR = "./model"
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    MODEL_NAME = "crop_disease_model.pth"
    CLASS_NAMES_FILE = "class_names.json"

# Load CLASS_NAMES from JSON file
if os.path.exists(Config.CLASS_NAMES_FILE):
    with open(Config.CLASS_NAMES_FILE, "r") as f:
        CLASS_NAMES = json.load(f)
else:
    CLASS_NAMES = []
    print(f"⚠️ Warning: {Config.CLASS_NAMES_FILE} not found. CLASS_NAMES is empty.")

# Treatment advice (keys must match CLASS_NAMES entries)
TREATMENTS = {
    "Healthy": "No treatment needed. Keep monitoring your plant.",
    "Powdery Mildew": "Remove affected leaves, improve air circulation, and use fungicide spray.",
    "Rust": "Remove and destroy infected leaves. Apply a copper-based fungicide.",
    "Leaf Spot": "Trim affected areas and avoid overhead watering."
}
