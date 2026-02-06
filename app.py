import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import time
import google.generativeai as genai

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="DermaScan AI | Professional Skin Analysis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. GEMINI AI CONFIGURATION
# ==========================================
# ⚠️ REPLACE THIS WITH YOUR REAL API KEY
GOOGLE_API_KEY = "API KEY GOES HERE"

# Setup the AI Model
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    model_ai = genai.GenerativeModel('gemini-2.5-flash')
except Exception as e:
    st.warning("⚠️ Chatbot disabled. Please add a valid Google API Key in line 26.")

# System Prompt to make the AI act like a medical assistant
SYSTEM_PROMPT = """
You are 'DermaScan AI', an advanced medical assistant capable of explaining skin lesion analysis.
Your goal is to assist users in understanding skin health, the technical details of this project, and the specific classes this model detects.

---
### 🧠 KNOWLEDGE BASE (STRICT FACTS)
1. **The Model:** This project uses a **ResNet18 (Convolutional Neural Network)** architecture.
2. **Optimization:** The model was optimized using **Quantization (FP16)** and **Pruning** to be lightweight (<25MB) for mobile deployment.
3. **Dataset:** Trained on the **HAM10000** dataset (Human Against Machine with 10,000 training images).
4. **Classes:** The model classifies skin lesions into exactly **7 CATEGORIES**:
   - **Melanocytic nevi (nv):** Common moles (usually benign).
   - **Melanoma (mel):** The most serious type of skin cancer.
   - **Basal cell carcinoma (bcc):** A common skin cancer.
   - **Actinic keratoses (akiec):** Pre-cancerous scaly patches.
   - **Benign keratosis-like lesions (bkl):** Non-cancerous growths (e.g., seborrheic keratosis).
   - **Dermatofibroma (df):** Benign skin nodules.
   - **Vascular lesions (vasc):** Blood vessel abnormalities (e.g., cherry angiomas).

---
### 🛡️ GUIDELINES & SAFETY
- **Diagnosis:** If a user asks "Do I have cancer?" or "Is this melanoma?", you MUST say: 
  *"I am an AI academic prototype. I cannot provide a medical diagnosis. Please consult a dermatologist immediately."*
- **Accuracy:** If asked about accuracy, state that the model achieves **~67% accuracy** on the test set, which is typical for this dataset due to class imbalance.
- **Tone:** Be empathetic, professional, and concise (keep answers under 3-4 sentences).
- **Context:** If asked "How does it work?", explain that it converts the image into tensors and looks for patterns using the ResNet18 layers.

---
### 🚫 WHAT TO AVOID
- Do NOT make up new skin diseases.
- Do NOT give treatment advice (e.g., "Use this cream").
- Do NOT say the model is 100% accurate.
"""

# ==========================================
# 3. CUSTOM CSS (Dark Mode & Mobile Fixes)
# ==========================================
st.markdown("""
    <style>
    /* Button Styling */
    div.stButton > button {
        background-color: #007bff;
        color: white !important;
        border-radius: 12px;
        border: none;
        padding: 12px 24px;
        font-weight: 600;
        width: 100%;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #0056b3;
        transform: translateY(-2px);
    }
    
    /* Metric Cards - Force Dark Text */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #e0e0e0;
    }
    div[data-testid="stMetric"] label {
        color: #6c757d !important; 
        font-size: 0.9rem;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #212529 !important;
        font-weight: 700;
    }

    /* Expander Styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        background-color: #f1f3f5;
        color: #212529 !important;
        border-radius: 8px;
    }
    
    /* Chat Message Styling */
    .stChatMessage {
        background-color: #f8f9fa;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    
    /* Mobile Media Query */
    @media only screen and (max-width: 600px) {
        div[data-testid="stMetricValue"] { font-size: 1.2rem !important; }
        div[data-testid="column"] { margin-bottom: 20px; }
    }
    </style>
""", unsafe_allow_html=True)

# Define Classes
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
# 4. MODEL LOADING (Safe Mode)
# ==========================================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 7)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "mini_project_2026_final.pth")
    
    try:
        model.half() 
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.float() 
        model.eval()
        return model
    except Exception as e:
        return None

model = load_model()

# ==========================================
# 5. IMAGE PREPROCESSING
# ==========================================
def process_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return preprocess(image).unsqueeze(0)

# ==========================================
# 6. UI LAYOUT
# ==========================================

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=80)
    st.title("DermaScan AI")
    st.caption("v2.1.0 | Mini Project 2026")
    st.markdown("---")
    
    st.header("Project Info")
    st.write("**Model:** ResNet18 (CNN)")
    st.write("**Accuracy:** ~67%")
    st.write("**Classes:** 7 Types")
    st.markdown("---")
    
    st.subheader("⚙️ How to Use")
    st.markdown("""
    1. **Upload** a clear image.
    2. **Analyze** the lesion.
    3. **Chat** with the AI Assistant below.
    """)
    
    st.markdown("---")
    st.warning("⚠️ **Disclaimer:** This tool is for **academic purposes only**. Do not use for real medical diagnosis.")
    st.info("🔒 **Privacy:** Images are processed locally.")

# Main Header (Centered Logo using HTML)
st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <img src="https://cdn-icons-png.flaticon.com/512/3004/3004458.png" width="120" style="margin-bottom: 10px;">
        <h1 style="color: #2c3e50; margin: 0;">AI-Powered Skin Analysis</h1>
        <h5 style="color: #6c757d; margin-top: 5px;">Advanced Dermatology Diagnostics using ResNet18</h5>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# Model Logic
if model is None:
    st.error("🚨 Model file not found. Please ensure 'mini_project_2026_final.pth' is uploaded.")
else:
    with st.expander("📂 Tap to Upload Image", expanded=True):
        uploaded_file = st.file_uploader("Choose the affected skin area...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
        # Responsive Columns
        col1, col2 = st.columns([1, 1.5], gap="large")
        
        with col1:
            st.markdown("#### 🖼️ Input Image")
            st.image(image, use_container_width=True, caption="Uploaded Scan")
        
        with col2:
            st.markdown("#### 📊 Analysis Results")
            
            if st.button("🔍 Analyze Lesion"):
                with st.spinner('Running Neural Network...'):
                    time.sleep(1) 
                    img_tensor = process_image(image)
                    with torch.no_grad():
                        outputs = model(img_tensor)
                        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                        confidence, predicted_class = torch.max(probabilities, 0)
                    
                    probs_np = probabilities.numpy()
                    sorted_indices = probs_np.argsort()[::-1]
                    
                    st.success("Analysis Complete")
                    
                    # Metrics
                    m1, m2 = st.columns(2)
                    m1.metric("Diagnosis", CLASSES[predicted_class.item()])
                    m2.metric("Confidence", f"{confidence.item()*100:.1f}%")
                    
                    st.markdown("---")
                    st.markdown("##### 📉 Top 3 Probabilities")
                    
                    for i in range(3):
                        idx = sorted_indices[i]
                        class_name = CLASSES[idx]
                        score = probs_np[idx]
                        
                        # HTML Progress Bars
                        st.markdown(f"""
                        <div style="margin-bottom: 12px;">
                            <div style="display:flex; justify-content:space-between; font-weight:500; font-size:0.95rem; margin-bottom:4px;">
                                <span>{class_name}</span>
                                <span>{score*100:.1f}%</span>
                            </div>
                            <div style="background-color: #e9ecef; border-radius: 6px; height: 10px; width: 100%; box-shadow: inset 0 1px 2px rgba(0,0,0,0.1);">
                                <div style="background-color: #007bff; height: 10px; border-radius: 6px; width: {score*100}%;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("👆 Click the button above to start the AI analysis.")

# ==========================================
# 7. CHATBOT SECTION (GEMINI API)
# ==========================================
st.markdown("---")
st.subheader("💬 AI Medical Assistant")
# st.caption("Powered by Google Gemini Pro - Ask anything!")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask about symptoms, precautions, or skin types..."):
    # Show User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate AI Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            if not GOOGLE_API_KEY or "YOUR_API_KEY" in GOOGLE_API_KEY:
                full_response = "⚠️ Please configure your Google API Key in the code to use the chatbot."
                message_placeholder.markdown(full_response)
            else:
                # Call Gemini
                response = model_ai.generate_content(SYSTEM_PROMPT + f"\nUser Question: {prompt}")
                bot_reply = response.text
                
                # Typing Effect
                for chunk in bot_reply.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
                
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
        except Exception as e:
            st.error(f"Error: {e}")