import streamlit as st
import numpy as np
from PIL import Image
import os

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
    try:
        import tensorflow as tf
        
        if not os.path.exists("cattle_breed_model.h5"):
            st.error("❌ Model file 'cattle_breed_model.h5' not found!")
            return None
            
        # Try different loading strategies
        try:
            # Method 1: Standard load
            model = tf.keras.models.load_model("cattle_breed_model.h5", compile=False)
            st.success("✅ Model loaded successfully with compile=False")
            return model
            
        except Exception as e1:
            st.warning(f"First load attempt failed: {e1}")
            
            try:
                # Method 2: Load with compilation
                model = tf.keras.models.load_model("cattle_breed_model.h5", compile=True)
                st.success("✅ Model loaded successfully with compile=True")
                return model
                
            except Exception as e2:
                st.warning(f"Second load attempt failed: {e2}")
                
                try:
                    # Method 3: Try loading with custom objects for specific architectures
                    # This is common for models with custom layers or functional API
                    model = tf.keras.models.load_model(
                        "cattle_breed_model.h5", 
                        compile=False,
                        custom_objects={}
                    )
                    st.success("✅ Model loaded with custom objects")
                    return model
                    
                except Exception as e3:
                    st.error(f"All loading methods failed: {e3}")
                    return None
                    
    except ImportError:
        st.error("❌ TensorFlow not available!")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error loading model: {e}")
        return None

# Load model
model = load_model()

# --- Prediction with error handling ---
def predict_breed(image):
    if model is None:
        return "Model not available", 0.0
    
    try:
        # Preprocess image
        image = image.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(image) / 255.0
        
        # Handle different input formats based on model architecture
        if len(model.input_shape) == 4:  # Standard CNN (batch, height, width, channels)
            img_array = np.expand_dims(img_array, axis=0)
        elif len(model.input_shape) == 2:  # Flattened input
            img_array = img_array.flatten()
            img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        prediction = model.predict(img_array, verbose=0)
        
        # Handle different output formats
        if isinstance(prediction, list):
            # Model returns multiple outputs
            prediction = prediction[0]  # Take the first output
        
        prediction = prediction[0]  # Get first batch element
        
        predicted_label = breed_labels[np.argmax(prediction)]
        confidence = float(np.max(prediction)) * 100
        return predicted_label, confidence
        
    except Exception as e:
        return f"Prediction error: {str(e)}", 0.0

# --- Streamlit UI ---
st.set_page_config(page_title="🐄 Cattle Breed Classifier", layout="centered")
st.title("🐄 Cattle Breed Classifier")
st.write("Upload a cattle image and let AI identify its breed.")

# Check TensorFlow availability
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
    tf_version = tf.__version__
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf_version = "Not installed"

# Display status
if not TENSORFLOW_AVAILABLE:
    st.error("❌ TensorFlow not installed!")
else:
    st.success(f"✅ TensorFlow {tf_version} installed")

if model is None:
    st.error("❌ Model could not be loaded. Possible reasons:")
    st.write("- Model architecture mismatch")
    st.write("- TensorFlow version incompatibility")
    st.write("- Corrupted model file")
    st.write("- Missing custom layers")
else:
    st.success("✅ Model loaded successfully!")
    
    # Display model info for debugging
    with st.expander("🔧 Model Information"):
        st.write(f"Input shape: {model.input_shape}")
        st.write(f"Output shape: {model.output_shape}")
        st.write(f"Number of layers: {len(model.layers)}")
        st.write(f"Model type: {type(model)}")

# File uploader
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
                st.info(f"Current confidence: {confidence:.2f}%")
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

# Debug information
with st.expander("🛠️ Debug Information"):
    st.write(f"TensorFlow Available: {TENSORFLOW_AVAILABLE}")
    if TENSORFLOW_AVAILABLE:
        st.write(f"TensorFlow Version: {tf_version}")
    st.write(f"Model Loaded: {model is not None}")
    st.write(f"Model File Exists: {os.path.exists('cattle_breed_model.h5')}")
    if os.path.exists("cattle_breed_model.h5"):
        st.write(f"Model file size: {os.path.getsize('cattle_breed_model.h5')} bytes")

# Troubleshooting guide
with st.expander("❓ Troubleshooting Guide"):
    st.write("""
    **Common Solutions:**
    
    1. **TensorFlow Version Mismatch:**
       - Try: `pip install tensorflow==2.12.0`
    
    2. **Model Architecture Issues:**
       - The model might have been created with a different architecture
       - Try recreating the model with current TensorFlow version
    
    3. **Custom Layers:**
       - If the model uses custom layers, they need to be defined during loading
    
    4. **Corrupted File:**
       - Verify the model file is not corrupted
       - Try re-uploading the model file
    
    **Quick Fix:**
    ```bash
    pip uninstall tensorflow -y
    pip install tensorflow==2.12.0
    ```
    """)
