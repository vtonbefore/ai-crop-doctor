import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

DATA_DIR = "./data/PlantVillage"   # Change if needed
MODEL_DIR = "./model"
MODEL_PATH = os.path.join(MODEL_DIR, "crop_disease_model.pth")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")

BATCH_SIZE = 16
NUM_EPOCHS = 2       # Short training
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_IMAGES = 500     # Limit dataset size for speed

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
    dataset_full = datasets.ImageFolder(DATA_DIR, transform=transform)

    # Save class names
    with open(CLASS_NAMES_PATH, "w") as f:
        json.dump(dataset_full.classes, f)

    # Take only a subset for speed
    indices = list(range(min(MAX_IMAGES, len(dataset_full))))
    dataset = Subset(dataset_full, indices)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    print(f"Training on {len(dataset)} images from {len(dataset_full.classes)} classes.")

    print("Building model...")
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(dataset_full.classes))
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
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

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
    print("✅ Training complete and model saved.")

if __name__ == "__main__":
    train()
