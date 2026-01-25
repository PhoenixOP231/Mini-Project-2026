import os
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ==========================================
# 1. CONFIGURATION & SETTINGS
# ==========================================
st.set_page_config(
    page_title="Skin Cancer Detection AI",
    page_icon="🩺",
    layout="centered"
)

# Define the exact class names (Alphabetical order, matching training)
CLASSES = [
    'Actinic keratoses (akiec)', 
    'Basal cell carcinoma (bcc)', 
    'Benign keratosis-like lesions (bkl)', 
    'Dermatofibroma (df)', 
    'Melanoma (mel)', 
    'Melanocytic nevi (nv)', 
    'Vascular lesions (vasc)'
]

# ==========================================
# 2. LOAD THE TRAINED MODEL
# ==========================================
@st.cache_resource
def load_model():
    # Initialize architecture
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 7)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "mini_project_2026_final.pth")
    
    try:
        # Load weights into a temporary 16-bit model to match the file
        model.half() 
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        
        # CONVERT BACK TO FLOAT32: This stops the RuntimeError on CPU
        model.float() 
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

try:
    model = load_model()
    if model is None:
        st.stop()
except Exception as e:
    st.error(f"Critical Error: {e}")
    st.stop()

# ==========================================
# 3. IMAGE PREPROCESSING
# ==========================================
def process_image(image):
    # Standard preprocessing for ResNet18
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return preprocess(image).unsqueeze(0)

# ==========================================
# 4. USER INTERFACE (UI)
# ==========================================
st.title("🩺 Skin Cancer Classification AI")
st.write("Upload a skin lesion image to analyze it using the Mini Project 2026 Model.")

# Sidebar for extra info
with st.sidebar:
    st.header("Project Info")
    st.write("**Model:** ResNet18 (CNN)")
    st.write("**Accuracy:** ~67%")
    st.write("**Classes:** 7 Types")
    st.info("Disclaimer: This is for academic purposes only. Do not use for real medical diagnosis.")

# File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Analysis Button
    if st.button("Analyze Lesion"):
        with st.spinner('Analyzing...'):
            # Preprocess
            img_tensor = process_image(image)
            
            # Predict
            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted_class = torch.max(probabilities, 0)
            
            # Show Result with Confidence Threshold
            threshold = 0.60  # You can adjust this (0.60 = 60%)
            
            if confidence.item() < threshold:
                st.warning(f"⚠️ Low Confidence ({confidence.item()*100:.2f}%).")
                st.write(f"The model thinks this is **{CLASSES[predicted_class.item()]}**, but is not sure.")
                st.info("Recommendation: Consult a dermatologist for a closer look.")
            else:
                st.success(f"**Prediction:** {CLASSES[predicted_class.item()]}")
                st.metric(label="Confidence Score", value=f"{confidence.item()*100:.2f}%")
            
            # Show probability chart
            st.write("---")
            st.write("**Detailed Probabilities:**")
            probs_dict = {CLASSES[i]: float(probabilities[i]) for i in range(7)}

            st.bar_chart(probs_dict)

