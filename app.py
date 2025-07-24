import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import json
import os

# Load model
model = tf.keras.models.load_model("model/plant_disease_model.h5")

# Load label map
with open("model/label_map.json", "r") as f:
    label_map = json.load(f)

# Reverse mapping: index → class name
index_to_class = {v: k for k, v in label_map.items()}

# Image preprocessing
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

# Disease tips (basic)
disease_info = {
    "Tomato___Early_blight": "Early blight can be treated with fungicides. Rotate crops yearly.",
    "Tomato___Late_blight": "Late blight spreads quickly. Remove infected plants immediately.",
    "Corn___Healthy": "Healthy leaf! Keep monitoring weekly.",
    "Potato___Early_blight": "Improve drainage and apply recommended treatment early.",
    # Add more classes and advice as needed...
}

# Streamlit UI
st.set_page_config(page_title="AI Plant Doctor", layout="centered")

st.title("🧠🌿 AI Plant Doctor")
st.write("Upload a leaf image to detect possible crop diseases.")

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Leaf Image', use_column_width=True)

    with st.spinner('Analyzing...'):
        processed = preprocess_image(image)
        prediction = model.predict(processed)
        pred_class_index = np.argmax(prediction)
        pred_class = index_to_class[pred_class_index]
        confidence = round(np.max(prediction) * 100, 2)

    st.success(f"🔍 Prediction: **{pred_class}** ({confidence}%)")

    if pred_class in disease_info:
        st.info(f"💡 Tip: {disease_info[pred_class]}")
    else:
        st.warning("No treatment info available for this class.")
