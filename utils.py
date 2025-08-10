import json
import os

def save_class_names(class_names, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(class_names, f)
    print(f"Class names saved to {path}")

def load_class_names(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Class names file not found: {path}")
    with open(path, 'r') as f:
        class_names = json.load(f)
    return class_names

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
