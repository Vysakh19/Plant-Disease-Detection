import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# -----------------------------
# 1. Paths
# -----------------------------
DATA_DIR = "data"
MODEL_DIR = "models"
LATEST_MODEL_PATH = os.path.join(MODEL_DIR, "model_latest.pth")

os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# 2. Device (GPU if available)
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# 3. Transforms
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -----------------------------
# 4. Dataset & DataLoader
# -----------------------------
dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    pin_memory=True if device.type == "cuda" else False
)

class_names = dataset.classes
num_classes = len(class_names)
print("Classes:", class_names)

# -----------------------------
# 5. Load Pretrained ResNet50
# -----------------------------
model = models.resnet50(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# -----------------------------
# 6. Resume if model exists
# -----------------------------
if os.path.exists(LATEST_MODEL_PATH):
    print("🔄 Loading existing model to resume training...")
    model.load_state_dict(torch.load(LATEST_MODEL_PATH, map_location=device))
else:
    print("🆕 No saved model found. Training from scratch.")

# -----------------------------
# 7. Loss & Optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# -----------------------------
# 8. Training Loop
# -----------------------------
epochs = 1  # epochs per run

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss:.4f}")

# -----------------------------
# 9. Save Model (Best Practice)
# -----------------------------

# Save as latest (for resume)
torch.save(model.state_dict(), LATEST_MODEL_PATH)

# Save versioned copy (history)
versioned_path = os.path.join(MODEL_DIR, f"model_epoch{epochs}.pth")
torch.save(model.state_dict(), versioned_path)

print("✅ Training completed")
print("✅ Latest model saved as:", LATEST_MODEL_PATH)
print("✅ Versioned model saved as:", versioned_path)
