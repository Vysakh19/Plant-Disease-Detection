import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
import numpy as np
import cv2
import os

# -----------------------------
# Paths
# -----------------------------
MODEL_PATH = "models/model_latest.pth"
DATA_DIR = "data"

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
# Load model (NO pretrained warning)
# -----------------------------
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# -----------------------------
# Grad-CAM hooks
# -----------------------------
gradients = None
activations = None

def backward_hook(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0]

def forward_hook(module, input, output):
    global activations
    activations = output

target_layer = model.layer3
target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)

# -----------------------------
# Prediction + Grad-CAM
# -----------------------------
def predict_and_gradcam(image):
    img_tensor = transform(image).unsqueeze(0).to(device)

    output = model(img_tensor)
    pred_class = output.argmax(dim=1)

    model.zero_grad()
    output[0, pred_class].backward()

    grads = gradients.cpu().data.numpy()[0]
    acts = activations.cpu().data.numpy()[0]

    weights = np.mean(grads, axis=(1,2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    cam = cv2.resize(cam, (224,224))

    img_cv = cv2.cvtColor(np.array(image.resize((224,224))), cv2.COLOR_RGB2BGR)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = heatmap * 0.4 + img_cv

    return class_names[pred_class.item()], overlay

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🌿 Plant Disease Detection with Grad-CAM")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", width=350)

    label, cam_image = predict_and_gradcam(image)

    st.success(f"Prediction: {label}")
    st.subheader("Grad-CAM (Highlighted Region)")
    st.image(cv2.cvtColor(cam_image.astype(np.uint8), cv2.COLOR_BGR2RGB),
             width=350)
