import os
import json
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# === Config ===
DATA_DIR = "./data/PlantVillage"  # Change to your test dataset path if different
MODEL_DIR = "./model"
MODEL_PATH = os.path.join(MODEL_DIR, "crop_disease_model.pth")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")

BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load class names ===
if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(f"Class names JSON not found at {CLASS_NAMES_PATH}")

with open(CLASS_NAMES_PATH, "r") as f:
    classes = json.load(f)

# === Data transforms ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === Load dataset ===
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# === Load model ===
model = models.resnet18(pretrained=False)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, len(classes))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# === Evaluate ===
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in dataloader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
