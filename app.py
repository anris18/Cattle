import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
import logging
logging.getLogger('streamlit').setLevel(logging.ERROR)

# Suppress specific Streamlit warnings
import sys
if not hasattr(sys, '_called_from_test'):
    # This prevents the ScriptRunContext warning
    import contextlib
    import io
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        import streamlit as st
else:
    import streamlit as st

import numpy as np
from PIL import Image

# Try to import TensorFlow with fallback
try:
    import tensorflow as tf
    # Suppress TensorFlow logging
    tf.get_logger().setLevel('ERROR')
    TENSORFLOW_AVAILABLE = True
except ImportError:
    st.error("TensorFlow is not installed. Please install it using: pip install tensorflow")
    TENSORFLOW_AVAILABLE = False

# --- Breed Information ---
breed_info_raw = {
    "ayrshire": """DEVELOPED IN THE COUNTY OF AYRSHIRE IN SOUTHWESTERN SCOTLAND
4500 Liters
BEST SUITED TO TEMPERATE CLIMATES
SCOTLAND
MEDIUM SIZE, REDDISH-BROWN AND WHITE SPOTS
8
ALERT AND ACTIVE
HIGH MILK QUALITY WITH GOOD FAT CONTENT""",

    "friesian": """ORIGINATING IN THE FRIESLAND REGION OF THE NETHERLANDS
6500 Liters
THRIVES IN TEMPERATE CLIMATES, REQUIRES HIGH-QUALITY FEED AND MANAGEMENT
NETHERLANDS
LARGE BODY SIZE, BLACK AND WHITE SPOTTED COAT
13
DOCILE, TOLERANT TO HARSH CONDITIONS
DUAL-PURPOSE: MILK AND DRAUGHT POWER""",

    "jersey": """BRITISH BREED, DEVELOPED IN JERSEY, CHANNEL ISLANDS
5500 Liters
THRIVES IN WARM CLIMATES, REQUIRES GOOD GRAZING PASTURES
SCOTLAND
SMALL TO MEDIUM BODY, LIGHT BROWN COLOR
10
DOCILE AND FRIENDLY
EFFICIENT MILK PRODUCTION WITH HIGH BUTTERFAT CONTENT""",

    "lankan white": """CROSSBREED BETWEEN ZEBU AND EUROPEAN BREEDS
4331 Liters
BEST SUITED TO TEMPERATE CLIMATES
SRI LANKA
MEDIUM-SIZED, ZEBU CHARACTERISTICS, HEAT TOLERANT
12
CALM BUT CAN BE AGGRESSIVE UNDER STRESS
HIGH MILK YIELD, SUITABLE FOR DAIRY FARMING""",

    "sahiwal": """ORIGINATING IN THE SAHIWAL DISTRICT OF PUNJAB, PAKISTAN
3000 Liters
ADAPTED TO TROPICAL CONDITIONS, HEAT-TOLERANT
PAKISTAN
MEDIUM SIZE, REDDISH BROWN COAT
6
CALM BUT CAN BE AGGRESSIVE UNDER STRESS
MODERATE MILK YIELD, RESISTANT TO DISEASE""",

    "zebu": """CROSSBREED BETWEEN ZEBU AND EUROPEAN BREEDS (AUSTRALIAN FRIESIAN)
4000 Liters
THRIVES IN TROPICAL CONDITIONS, HIGH RESISTANCE TO HEAT
AUSTRALIA
MEDIUM-SIZED, ZEBU CHARACTERISTICS, HEAT TOLERANCE
10
DOCILE
MODERATE MILK YIELD, RESISTANT TO DISEASE"""
}

breed_info = {k.lower().strip(): v for k, v in breed_info_raw.items()}
breed_labels = ["Ayrshire", "Friesian", "Jersey", "Lankan White", "Sahiwal", "Zebu"]

IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 60.0

# --- Load Model ---
@st.cache_resource
def load_model():
    if not TENSORFLOW_AVAILABLE:
        return None
        
    try:
        if os.path.exists("cattle_breed_model.h5"):
            # Try multiple loading approaches
            try:
                return tf.keras.models.load_model("cattle_breed_model.h5", compile=False)
            except Exception as e:
                st.warning(f"First load attempt failed: {e}")
                try:
                    return tf.keras.models.load_model("cattle_breed_model.h5", compile=True)
                except Exception as e:
                    st.warning(f"Second load attempt failed: {e}")
                    return None
        else:
            st.error("❌ Model file 'cattle_breed_model.h5' not found!")
            return None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# Load model
model = load_model() if TENSORFLOW_AVAILABLE else None

# --- Prediction ---
def predict_breed(image):
    if model is None:
        return "Model not available", 0.0
    
    try:
        image = image.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array, verbose=0)[0]
        predicted_label = breed_labels[np.argmax(prediction)]
        confidence = float(np.max(prediction)) * 100
        return predicted_label, confidence
    except Exception as e:
        return f"Prediction error: {e}", 0.0

# --- Streamlit UI ---
st.set_page_config(page_title="🐄 Cattle Breed Classifier", layout="centered")
st.title("🐄 Cattle Breed Classifier")
st.write("Upload a cattle image and let AI identify its breed.")

# Display status
if not TENSORFLOW_AVAILABLE:
    st.error("❌ TensorFlow not installed! Run: `pip install tensorflow`")
elif model is None:
    st.warning("⚠️ Model not loaded. Check if 'cattle_breed_model.h5' exists in the same folder.")
else:
    st.success("✅ Model loaded successfully!")

uploaded_file = st.file_uploader("Upload Cattle Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="📷 Uploaded Cattle Image", use_column_width=True)

        if model is not None:
            with st.spinner("🔍 Identifying breed..."):
                breed, confidence = predict_breed(image)

            if "error" in breed.lower():
                st.error(f"❌ {breed}")
            elif confidence < CONFIDENCE_THRESHOLD:
                st.error("🚫 Low confidence. Try a clearer cattle image.")
                st.info(f"Current confidence: {confidence:.2f}% (needs > {CONFIDENCE_THRESHOLD}%)")
            else:
                st.success(f"✅ Predicted Breed: **{breed}**")
                st.info(f"🔎 Confidence: {confidence:.2f}%")

                breed_key = breed.lower().strip()
                if breed_key in breed_info:
                    lines = breed_info[breed_key].strip().split("\n")
                    if len(lines) >= 8:
                        st.subheader("📚 Breed Information")
                        st.write(f"🧬 **Pedigree / Lineage:** {lines[0]}")
                        st.write(f"🍼 **Productivity:** {lines[1]}")
                        st.write(f"🌿 **Optimal Rearing Conditions:** {lines[2]}")
                        st.write(f"🌍 **Origin:** {lines[3]}")
                        st.write(f"🐮 **Physical Characteristics:** {lines[4]}")
                        st.write(f"❤️️ **Lifespan (Years):** {lines[5]}")
                        st.write(f"💉 **Temperament:** {lines[6]}")
                        st.write(f"🥩 **Productivity Metrics:** {lines[7]}")
                else:
                    st.warning("⚠️ No additional information found for this breed.")
        else:
            st.warning("⚠️ Cannot make predictions - model is not loaded.")
            
    except Exception as e:
        st.error(f"❌ Error processing image: {e}")

# Debug information (collapsed by default)
with st.expander("🛠️ Debug Information"):
    st.write(f"TensorFlow Available: {TENSORFLOW_AVAILABLE}")
    st.write(f"Model Loaded: {model is not None}")
    if os.path.exists("cattle_breed_model.h5"):
        st.write(f"Model file size: {os.path.getsize('cattle_breed_model.h5')} bytes")
    else:
        st.write("Model file: Not found")
    
    if TENSORFLOW_AVAILABLE:
        st.write(f"TensorFlow Version: {tf.__version__}")

# Instructions
with st.expander("💡 Setup Instructions"):
    st.write("""
    **To run this app:**
    
    1. **Install dependencies:**
    ```bash
    pip install tensorflow streamlit pillow numpy
    ```
    
    2. **Place 'cattle_breed_model.h5' in the same folder**
    
    3. **Run:**
    ```bash
    streamlit run app.py
    ```
    
    **If you still see warnings:** They are harmless and don't affect functionality!
    """)
