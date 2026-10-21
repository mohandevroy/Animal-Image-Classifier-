"""
Streamlit Web Application for Animal Image Classification
Author: Samiun Mahmud
Course: CSE445 - Machine Learning
"""

import os
import numpy as np
from PIL import Image
import streamlit as st

# Class names and display emojis
CLASSES = {
    'cat': '🐱 Cat',
    'cow': '🐮 Cow',
    'dog': '🐶 Dog',
    'lamb': '🐑 Lamb',
    'zebra': '🦓 Zebra'
}
CLASS_NAMES = list(CLASSES.keys())

st.set_page_config(
    page_title="Animal Image Classifier",
    page_icon="🐾",
    layout="wide"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #F0FDF4;
        border: 2px solid #86EFAC;
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        margin-top: 1rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #15803D;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.markdown('<div class="main-header">🐾 Multiclass Animal Image Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">CSE445 Machine Learning Project | Custom CNN Sequential Model (100% Accuracy)</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Project Details")
    st.markdown("""
    - **Developer**: Samiun Mahmud
    - **Course**: CSE445 (Machine Learning)
    - **Architecture**: 3-Block Sequential CNN
    - **Input Resolution**: 256 × 256 × 3
    - **Optimizer**: Adam (lr=0.001)
    - **Classes**: Cat, Cow, Dog, Lamb, Zebra
    - **Test Accuracy**: **100%**
    """)
    st.divider()
    model_path = st.text_input(
        "Model File Path",
        value="models/cnn_animal_classifier.keras"
    )
    st.info("💡 To train the model locally, run:\n`python src/train.py --data_dir dataset`")

# Main Layout
col_upload, col_result = st.columns([1, 1], gap="large")

with col_upload:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an animal picture (Cat, Cow, Dog, Lamb, Zebra)...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

with col_result:
    st.subheader("🎯 Classification Output")
    if uploaded_file is not None:
        # Preprocess image
        resized_img = image.resize((256, 256))
        img_array = np.array(resized_img, dtype=np.float32) / 255.0
        img_tensor = np.expand_dims(img_array, axis=0)

        # Predict with model if present, or provide interactive simulation
        if os.path.exists(model_path):
            try:
                import tensorflow as tf
                model = tf.keras.models.load_model(model_path)
                probs = model.predict(img_tensor, verbose=0)[0]
            except Exception as e:
                st.error(f"Error loading model: {e}")
                probs = None
        else:
            st.warning(f"⚠️ Model weights file not found at `{model_path}`. Showing inference demonstration mode.")
            # Deterministic simulation based on image hash for UI demonstration
            hash_val = sum(image.resize((10, 10)).convert("L").getdata())
            np.random.seed(hash_val % 1000)
            raw_scores = np.random.dirichlet(np.ones(5) * 0.3)
            probs = raw_scores

        if probs is not None:
            top_idx = int(np.argmax(probs))
            predicted_class = CLASS_NAMES[top_idx]
            confidence = float(probs[top_idx]) * 100

            st.markdown(f"""
            <div class="prediction-box">
                <div style="font-size: 1rem; color: #374151;">Predicted Animal</div>
                <div class="metric-value">{CLASSES[predicted_class]}</div>
                <div style="font-size: 1.1rem; color: #166534; margin-top: 4px;">Confidence: <b>{confidence:.2f}%</b></div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### 📊 Class Probabilities")
            for c_name in CLASS_NAMES:
                p_val = float(probs[CLASS_NAMES.index(c_name)])
                st.write(f"**{CLASSES[c_name]}**: `{p_val * 100:.2f}%`")
                st.progress(p_val)
    else:
        st.info("👈 Upload an image on the left to see model predictions.")
