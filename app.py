import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import timm
import matplotlib.pyplot as plt
import numpy as np
import captum.attr as attr
import os

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['Aphids_Disease', 'Blotch', 'Healthy_Leaf', 'Leaf_Spot']

# Load Model
@st.cache_resource
def load_model():
    model = timm.create_model('mobilevitv2_200', pretrained=True, num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load('turmeric_mobilevitv2_200.pth', map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()


def apply_gradcam(model, image_tensor):
    grad_cam = attr.LayerGradCam(model, model.conv_stem) 
    attributions = grad_cam.attribute(image_tensor, target=0)
    return attributions.squeeze().cpu().detach().numpy()

# Image Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Streamlit UI
st.title("Turmeric MobileViTv2 Image Classifier with Grad-CAM")
st.write("Upload an image to classify with MobileViTv2 and explain predictions using Grad-CAM.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        output = model(input_tensor)
        _, pred_class = torch.max(output, 1)
        class_name = CLASS_NAMES[pred_class.item()]
    
    st.write(f"Predicted Class: **{class_name}**")
    
    # Grad-CAM
    cam_attributions = apply_gradcam(model, input_tensor)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.imshow(cam_attributions, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.title("Grad-CAM Heatmap")
    st.pyplot(plt)
