import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# -----------------------------
# Config
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
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -----------------------------
# Load class names
# -----------------------------
class_names = sorted(os.listdir(DATA_DIR))
num_classes = len(class_names)

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    return model

model = load_model()

# -----------------------------
# UI
# -----------------------------
st.title("🌿 Plant Disease Detection")
st.write("Upload a potato leaf image to detect disease")

uploaded_file = st.file_uploader("Upload leaf image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=500)

    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        _, pred = torch.max(outputs, 1)

    st.success(f"Prediction: **{class_names[pred.item()]}**")
