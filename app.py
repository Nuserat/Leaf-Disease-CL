import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import timm
import matplotlib.pyplot as plt
import numpy as np
import captum.attr as attr

# Load Model
@st.cache_resource
def load_model():
    model = timm.create_model('mobilevitv2_200', pretrained=False, num_classes=1000)
    model.load_state_dict(torch.load("model.h5", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

def apply_gradcam(model, image_tensor):
    grad_cam = attr.LayerGradCam(model, model.conv_stem)
    attributions = grad_cam.attribute(image_tensor, target=0)
    return attributions.squeeze().cpu().numpy()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Streamlit UI
st.title("MobileViTv2 Image Classifier with XAI")
st.write("Upload an image to classify using MobileViTv2 with explainability (Grad-CAM).")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess image
    input_tensor = transform(image).unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_class = torch.max(output, 1)
    
    st.write(f"Predicted Class: {predicted_class.item()}")
    
    # Apply Grad-CAM
    cam_attributions = apply_gradcam(model, input_tensor)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(cam_attributions, cmap='jet', alpha=0.5)
    plt.colorbar()
    plt.title("Grad-CAM Explanation")
    st.pyplot(plt)

# Plot training and validation loss
if st.button("Show Training Loss Curve"):
    train_losses = np.random.rand(50)  # Placeholder values
    val_losses = np.random.rand(50)  # Placeholder values
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 51), train_losses, label='Training Loss')
    plt.plot(range(1, 51), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
