import torch
from torchvision import transforms, models
from PIL import Image
import json
import os
import sys

MODEL_DIR = "./model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if len(sys.argv) != 2:
    print("Usage: python inference.py <image_path>")
    sys.exit()

image_path = sys.argv[1]

# Load detailed class info
with open(os.path.join(MODEL_DIR, "class_names.json"), "r") as f:
    class_info_list = json.load(f)

classes = [entry["class"] for entry in class_info_list]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(DEVICE)

model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "crop_disease_model.pth"), map_location=DEVICE))
model.to(DEVICE)
model.eval()

with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output, 1)
    pred_idx = predicted.item()
    pred_class_info = class_info_list[pred_idx]

print(f"Prediction: {pred_class_info['class']}")
print(f"Description: {pred_class_info['description']}")
print(f"Treatment: {pred_class_info['treatment']}")
print(f"Prevention: {pred_class_info['prevention']}")
