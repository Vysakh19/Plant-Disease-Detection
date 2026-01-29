import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# -----------------------------
# 1. Dataset path
# -----------------------------
DATA_DIR = "data"

# -----------------------------
# 2. Device (GPU first, else CPU)
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# 3. Transforms (Resize + Normalize)
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -----------------------------
# 4. Load Dataset using ImageFolder
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

print("Classes found:", class_names)
print("Number of classes:", num_classes)

# -----------------------------
# 5. Load Pretrained ResNet50
# -----------------------------
model = models.resnet50(pretrained=True)

# -----------------------------
# 6. Freeze all layers
# -----------------------------
for param in model.parameters():
    param.requires_grad = False

# -----------------------------
# 7. Replace final layer
# -----------------------------
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Move model to device (GPU/CPU)
model = model.to(device)

# -----------------------------
# 8. Test one batch (sanity check)
# -----------------------------
images, labels = next(iter(dataloader))
images = images.to(device, non_blocking=True)
labels = labels.to(device, non_blocking=True)

outputs = model(images)
print("Output shape:", outputs.shape)

print("✅ DataLoader works")
print("✅ Model loads correctly on", device)
print("✅ Ready for training")
