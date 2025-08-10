import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

DATA_DIR = "./data/PlantVillage"   
MODEL_DIR = "./model"
MODEL_PATH = os.path.join(MODEL_DIR, "crop_disease_model.pth")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")

BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        print(f"Created model directory at {MODEL_DIR}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    print(f"Loading dataset from {DATA_DIR} ...")
    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

    if not os.path.exists(CLASS_NAMES_PATH):
        print(f"Saving class names to {CLASS_NAMES_PATH} ...")
        with open(CLASS_NAMES_PATH, "w") as f:
            json.dump(dataset.classes, f)
    else:
        print(f"Class names JSON already exists at {CLASS_NAMES_PATH}")

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    print("Building model...")
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(dataset.classes))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Starting training for {NUM_EPOCHS} epochs on device {DEVICE} ...")

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in dataloader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = 100 * correct / total
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.2f}%")

    print(f"Saving model to {MODEL_PATH} ...")
    torch.save(model.state_dict(), MODEL_PATH)
    print("Training complete and model saved.")

if __name__ == "__main__":
    train()
