import os
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms

# ======================
# SAFETY CHECKS
# ======================
missing_files = []
for f in ["inference.py", "model.py", "config.py"]:
    if not os.path.exists(f):
        missing_files.append(f)

# Create uploads folder if missing
if not os.path.exists("uploads"):
    os.makedirs("uploads")

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="🌱 AI Crop Doctor",
    page_icon="🌿",
    layout="wide"
)

# ======================
# HEADER
# ======================
st.markdown(
    """
    <style>
    .main-title { font-size: 42px; font-weight: 700; color: #2E7D32; }
    .sub-title { font-size: 18px; color: #555; }
    .footer { font-size: 12px; color: gray; text-align: center; margin-top: 50px; }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<p class="main-title">🌱 AI-Powered Crop Doctor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Detect crop leaf diseases instantly and get treatment advice.</p>', unsafe_allow_html=True)
st.write("---")

# Show missing files warning
if missing_files:
    st.error(f"⚠ Missing required files: {', '.join(missing_files)}")
    st.stop()

# ======================
# IMPORT AFTER CHECKS
# ======================
from config import CLASS_NAMES, TREATMENTS
from model import load_model

# ======================
# SIDEBAR
# ======================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2909/2909767.png", width=120)
st.sidebar.title("About")
st.sidebar.write("""
This AI model detects plant diseases from leaf images and provides treatment guidance.
It’s trained on a curated dataset of healthy and diseased leaves.
""")
st.sidebar.write("**Classes:**")
for c in CLASS_NAMES:
    st.sidebar.write(f"- {c}")
st.sidebar.info("Built with ❤️ using PyTorch + Streamlit")

# ======================
# MODEL LOADING
# ======================
@st.cache_resource
def load_ai_model():
    model = model
    model.load_states_dict(torch.load("plant_disease_model.pth"))
    model.eval()
    return model

with st.spinner("Loading AI model... Please wait ⏳"):
    try:
        model = load_ai_model()
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

st.success("✅ AI model loaded successfully!")


# ======================
# UPLOAD SECTION
# ======================
st.header("📤 Upload a Leaf Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Show image preview
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption="Uploaded Leaf", use_container_width=True)

        # Save temp image
        save_path = "uploads/temp.jpg"
        image.save(save_path)

        # Transform and predict
        with st.spinner("Analyzing leaf..."):
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
            img_t = transform(image).unsqueeze(0)
            with torch.no_grad():
                outputs = model(img_t)
                probs = torch.nn.functional.softmax(outputs, dim=1)[0]
                predicted_idx = torch.argmax(probs).item()
                predicted_class = CLASS_NAMES[predicted_idx]

        # Display results
        with col2:
            st.success(f"**Prediction:** {predicted_class}")
            st.write(f"**Confidence:** {probs[predicted_idx]*100:.2f}%")
            st.info(f"**Treatment Advice:** {TREATMENTS[predicted_class]}")

        # Confidence chart
        fig, ax = plt.subplots()
        ax.barh(CLASS_NAMES, probs.tolist(), color='green')
        ax.set_xlabel("Confidence")
        ax.set_xlim(0, 1)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")

else:
    st.warning("Please upload a leaf image to start.")

# ======================
# FOOTER
# ======================
st.markdown('<p class="footer">AI Crop Doctor © 2025 | Developed for academic purposes</p>', unsafe_allow_html=True)
