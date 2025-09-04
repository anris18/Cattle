import streamlit as st
import numpy as np
from PIL import Image
import os

# Check if model files exist
model_files = ["cattle_breed_model.h5", "cattle_breed_model.tflite"]
model_available = any(os.path.exists(f) for f in model_files)

# Try to import TensorFlow only if model files exist
if model_available:
    try:
        import tensorflow as tf
        TF_AVAILABLE = True
    except:
        TF_AVAILABLE = False
        st.warning("TensorFlow is not available. Running in demo mode.")
else:
    TF_AVAILABLE = False

# Load model if available
@st.cache_resource
def load_model():
    if not model_available:
        return None, "none"
    
    model = None
    model_type = "none"
    
    # Try to load TensorFlow model
    if TF_AVAILABLE:
        try:
            if os.path.exists("cattle_breed_model.h5"):
                model = tf.keras.models.load_model("cattle_breed_model.h5")
                model_type = "tf"
                st.success("✅ Loaded TensorFlow model")
            elif os.path.exists("cattle_breed_model.tflite"):
                # For TensorFlow Lite
                import tflite_runtime.interpreter as tflite
                interpreter = tflite.Interpreter(model_path="cattle_breed_model.tflite")
                interpreter.allocate_tensors()
                model = interpreter
                model_type = "tflite"
                st.success("✅ Loaded TensorFlow Lite model")
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            return None, "none"
    
    return model, model_type

model, model_type = load_model()

# Breed information (same as before)
breed_info_raw = {
    "ayrshire": """DEVELOPED IN THE COUNTY OF AYRSHIRE IN SOUTHWESTERN SCOTLAND
4500 Liters
BEST SUITED TO TEMPERATE CLIMATES
SCOTLAND
MEDIUM SIZE, REDDISH-BROWN AND WHITE SPOTS
8
ALERT AND ACTIVE
HIGH MILK QUALITY WITH GOOD FAT CONTENT""",
    # ... (other breeds remain the same)
}

# Normalize keys
breed_info = {k.lower().strip(): v for k, v in breed_info_raw.items()}

# Breed labels
breed_labels = ["Ayrshire", "Friesian", "Jersey", "Lankan White", "Sahiwal", "Zebu"]

IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 60.0

# Streamlit UI
st.set_page_config(page_title="🐄 Cattle Breed Identifier", layout="centered")
st.title("🐄 Cattle Breed Identifier")
st.write("Upload an image of a cow to predict its breed.")

if not model_available:
    st.info("🔧 Running in demonstration mode. Upload a model file for real predictions.")

# Image uploader
uploaded_file = st.file_uploader("Choose a cattle image", type=["jpg", "jpeg", "png"])

# Prediction function
def predict_breed(image):
    # Preprocess image
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    if model is not None and model_type == "tf":
        # Real prediction with TensorFlow model
        prediction = model.predict(img_array)[0]
    else:
        # Demo mode - return weighted random prediction (favor certain breeds)
        weights = np.array([0.15, 0.25, 0.20, 0.10, 0.15, 0.15])  # Weighted probabilities
        prediction = np.random.dirichlet(weights * 10)  # More realistic distribution
    
    predicted_label = breed_labels[np.argmax(prediction)]
    confidence = float(np.max(prediction)) * 100
    return predicted_label, confidence

def display_breed_info(breed_key, raw_text):
    try:
        lines = raw_text.strip().split("\n")
        if len(lines) < 8:
            st.warning("⚠️ Incomplete breed info.")
            return

        info_html = f"""
        <div style="
            border: 2px solid #007bff; 
            background-color: #e7f1ff; 
            padding: 15px; 
            border-radius: 8px; 
            margin-bottom: 10px;
            font-size: 16px;
        ">
            <p>🧬 <b>Pedigree / Lineage</b>: {lines[0]}</p>
            <p>🍼 <b>Productivity</b>: {lines[1]}</p>
            <p>🌿 <b>Optimal Rearing Conditions</b>: {lines[2]}</p>
            <p>🌍 <b>Origin</b>: {lines[3]}</p>
            <p>🐮 <b>Physical Characteristics</b>: {lines[4]}</p>
            <p>❤️️ <b>Lifespan (Years)</b>: {lines[5]}</p>
            <p>💉 <b>Temperament</b>: {lines[6]}</p>
            <p>🥩 <b>Productivity Metrics</b>: {lines[7]}</p>
        </div>
        """
        st.markdown(info_html, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"❌ Error parsing breed info: {str(e)}")

# Handle image and prediction
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='📷 Uploaded Cattle Image', use_container_width=True)

        with st.spinner("🔍 Identifying breed..."):
            breed, confidence = predict_breed(image)

        if model is None:
            st.info("🎭 Demonstration mode - showing sample prediction")
        
        if confidence < CONFIDENCE_THRESHOLD:
            st.error("🚫 Could not confidently identify the breed. Try another or clearer image.")
        else:
            st.success(f"✅ Predicted Breed: **{breed}**")
            st.caption(f"🔎 Confidence: {confidence:.2f}%")

            breed_key = breed.lower().strip()
            if breed_key in breed_info:
                st.subheader("📚 Breed Information")
                display_breed_info(breed_key, breed_info[breed_key])
            else:
                st.warning("⚠️ No additional information found for this breed.")

    except Exception as e:
        st.error(f"⚠️ Error processing image: {str(e)}")
