import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import os

# -----------------------------
# Paths
# -----------------------------python -c "import cv2; print(cv2.__version__)"
MODEL_PATH = "models/model_latest.pth"
DATA_DIR = "data"
IMAGE_PATH = "testimages/test2.JPG"

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Transforms
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# -----------------------------
# Load class names
# -----------------------------
class_names = sorted(os.listdir(DATA_DIR))
num_classes = len(class_names)

# -----------------------------
# Load model
# -----------------------------
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# -----------------------------
# Hook for Grad-CAM
# -----------------------------
gradients = None
activations = None

def backward_hook(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0]

def forward_hook(module, input, output):
    global activations
    activations = output

# Register hooks on last conv layer
target_layer = model.layer4
target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)

# -----------------------------
# Load image
# -----------------------------
image = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

# -----------------------------
# Forward pass
# -----------------------------
output = model(input_tensor)
pred_class = output.argmax(dim=1)

# -----------------------------
# Backward pass
# -----------------------------
model.zero_grad()
output[0, pred_class].backward()

# -----------------------------
# Generate heatmap
# -----------------------------
grads = gradients.cpu().data.numpy()[0]
acts = activations.cpu().data.numpy()[0]

weights = np.mean(grads, axis=(1,2))
cam = np.zeros(acts.shape[1:], dtype=np.float32)

for i, w in enumerate(weights):
    cam += w * acts[i]

cam = np.maximum(cam, 0)
cam = cam / cam.max()
cam = cv2.resize(cam, (224,224))

# -----------------------------
# Overlay heatmap
# -----------------------------
img = cv2.imread(IMAGE_PATH)
img = cv2.resize(img, (224,224))

heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
overlay = heatmap * 0.4 + img

cv2.imwrite("gradcam_result.jpg", overlay)

print("Grad-CAM saved as gradcam_result.jpg")
print("Predicted class:", class_names[pred_class.item()])
