import streamlit as st
import numpy as np
from PIL import Image
import cv2

# For TensorFlow Lite version (more compatible)
try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
except:
    TFLITE_AVAILABLE = False

# For full TensorFlow version
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except:
    TF_AVAILABLE = False

# Load model - try multiple approaches for compatibility
@st.cache_resource
def load_model():
    model = None
    model_type = "none"
    
    # Try to load TensorFlow Lite model first (more compatible)
    if TFLITE_AVAILABLE:
        try:
            interpreter = tflite.Interpreter(model_path="cattle_breed_model.tflite")
            interpreter.allocate_tensors()
            model = interpreter
            model_type = "tflite"
            st.success("Loaded TensorFlow Lite model")
        except:
            pass
    
    # Try to load full TensorFlow model if Lite not available
    if model is None and TF_AVAILABLE:
        try:
            model = tf.keras.models.load_model("cattle_breed_model.h5")
            model_type = "tf"
            st.success("Loaded TensorFlow model")
        except:
            pass
    
    return model, model_type

model, model_type = load_model()

# Breed information split into fields
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

# Normalize keys
breed_info = {k.lower().strip(): v for k, v in breed_info_raw.items()}

# Breed labels in model output order
breed_labels = ["Ayrshire", "Friesian", "Jersey", "Lankan White", "Sahiwal", "Zebu"]

IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 60.0

# Streamlit UI setup
st.set_page_config(page_title="🐄 Cattle Breed Identifier", layout="centered")
st.title("🐄 Cattle Breed Identifier")
st.write("Upload an image of a cow to predict its breed.")

# Check if model loaded successfully
if model is None:
    st.error("❌ Could not load the model. Please ensure the model file is available.")
    st.info("If you don't have a model, the app will run in demonstration mode with sample predictions.")
else:
    st.info("📁 Please upload a cattle image to start prediction.")

# Image uploader
uploaded_file = st.file_uploader("Choose a cattle image", type=["jpg", "jpeg", "png"])

# Prediction function
def predict_breed(image):
    # Preprocess image
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    if model_type == "tflite":
        # Get input and output tensors
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        
        # Set the tensor to point to the input data
        model.set_tensor(input_details[0]['index'], img_array.astype(np.float32))
        
        # Run the computation
        model.invoke()
        
        # Extract the output
        prediction = model.get_tensor(output_details[0]['index'])[0]
    elif model_type == "tf":
        prediction = model.predict(img_array)[0]
    else:
        # Demo mode - return random prediction
        prediction = np.random.rand(len(breed_labels))
        prediction = prediction / np.sum(prediction)  # Normalize to sum to 1
    
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
            st.warning("⚠️ Running in demonstration mode (no actual model loaded)")
        
        if confidence < CONFIDENCE_THRESHOLD:
            st.error("🚫 Could not confidently identify the breed. Try another or clearer image.")
        else:
            st.success(f"✅ Predicted Breed: **{breed}**")
            st.caption(f"🔎 Confidence: {confidence:.2f}%")

            breed_key = breed.lower().strip()
            if breed_key in breed_info:
                st.subheader("📚 Structured Breed Information")
                display_breed_info(breed_key, breed_info[breed_key])
            else:
                st.warning("⚠️ No additional information found for this breed.")

    except Exception as e:
        st.error(f"⚠️ Error processing image: {str(e)}")
