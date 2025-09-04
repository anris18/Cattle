import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- Load Model ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cattle_breed_model.h5", compile=False)

model = load_model()

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

# --- Prediction ---
def predict_breed(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array, verbose=0)[0]
    predicted_label = breed_labels[np.argmax(prediction)]
    confidence = float(np.max(prediction)) * 100
    return predicted_label, confidence

# --- Streamlit UI ---
st.set_page_config(page_title="🐄 Cattle Breed Classifier", layout="centered")
st.title("🐄 Cattle Breed Classifier")
st.write("Upload a cattle image and let AI identify its breed.")

uploaded_file = st.file_uploader("Upload Cattle Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Uploaded Cattle Image", use_column_width=True)

    st.write("🔍 Identifying breed...")
    breed, confidence = predict_breed(image)

    if confidence < CONFIDENCE_THRESHOLD:
        st.error("🚫 Could not confidently identify the breed. Try another or clearer image.")
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
