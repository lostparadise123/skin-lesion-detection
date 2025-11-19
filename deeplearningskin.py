import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

# ----------------------------
# Load Trained Model
# ----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('fine_tuned_densenet121.h5')
    return model

model = load_model()

# ----------------------------
# App Page Config
# ----------------------------
st.set_page_config(page_title="Skin Cancer Detection", page_icon="🩺", layout="centered")

# ----------------------------
# Custom CSS Styling
# ----------------------------
st.markdown("""
<style>
    body {
        background-color: #f7f9fb;
        color: #333333;
        font-family: "Segoe UI", sans-serif;
    }
    .main-title {
        background: linear-gradient(90deg, #2193b0, #6dd5ed);
        color: white;
        text-align: center;
        padding: 1.2rem;
        border-radius: 10px;
        font-size: 28px;
        font-weight: 700;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }
    .subtitle {
        text-align: center;
        font-size: 17px;
        color: #444;
        margin-top: 5px;
        margin-bottom: 25px;
    }
    .result-box {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .confidence-text {
        text-align: center;
        font-weight: 500;
        font-size: 16px;
        margin-top: 5px;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Title Section
# ----------------------------
st.markdown('<div class="main-title">🩺 Skin Cancer Classification Tool</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered detection of benign and malignant skin lesions</div>', unsafe_allow_html=True)

# ----------------------------
# Upload Section
# ----------------------------
uploaded_file = st.file_uploader(" Upload a skin lesion image (JPG or PNG):", type=None)

# ----------------------------
# Prediction Logic
# ----------------------------
if uploaded_file is not None:
    # Preprocess the image
    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]
    label = "Malignant" if prediction > 0.5 else "Benign"
    confidence = prediction if prediction > 0.5 else (1 - prediction)

    # ----------------------------
    # Result Display
    # ----------------------------
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    
    if label == "Malignant":
        st.error("Prediction: Malignant Lesion,need proper medical")
        bar_color = "#FF4B4B"  # red
    else:
        st.success("Prediction: Benign Lesion")
        bar_color = "#00C851"  # green
    
    st.markdown("</div>", unsafe_allow_html=True)

    # Custom confidence progress bar
    st.write("### Confidence Level:")
    progress_html = f"""
        <div style='background-color:#eee; border-radius:20px; height:25px; width:100%;'>
            <div style='background-color:{bar_color}; width:{confidence*100:.2f}%; height:25px; border-radius:20px; text-align:center; color:white; font-weight:600;'>
                {confidence*100:.2f}%
            </div>
        </div>
    """
    st.markdown(progress_html, unsafe_allow_html=True)
    st.markdown("<p class='confidence-text'>Confidence </p>", unsafe_allow_html=True)

    