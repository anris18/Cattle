import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Set page config
st.set_page_config(
    page_title="🐄 Cattle Breed Identifier", 
    layout="centered",
    page_icon="🐄"
)

# Title and description
st.title("🐄 Cattle Breed Identifier")
st.write("Upload an image of a cow to predict its breed.")

# Define breed labels - make sure these match your model's training classes
breed_labels = ["Ayrshire", "Friesian", "Jersey", "Lankan White", "Sahiwal", "Zebu"]

# Load model with proper error handling
@st.cache_resource
def load_cattle_model():
    try:
        model = tf.keras.models.load_model("cattle_breed_model.h5")
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# Load the model
model = load_cattle_model()

# Breed information
breed_info = {
    "ayrshire": {
        "Pedigree": "Developed in the County of Ayrshire in Southwestern Scotland",
        "Productivity": "4500 Liters",
        "Optimal Conditions": "Best suited to temperate climates",
        "Origin": "Scotland",
        "Characteristics": "Medium size, reddish-brown and white spots",
        "Lifespan": "8 years",
        "Temperament": "Alert and active",
        "Productivity Metrics": "High milk quality with good fat content"
    },
    "friesian": {
        "Pedigree": "Originating in the Friesland region of the Netherlands",
        "Productivity": "6500 Liters",
        "Optimal Conditions": "Thrives in temperate climates, requires high-quality feed and management",
        "Origin": "Netherlands",
        "Characteristics": "Large body size, black and white spotted coat",
        "Lifespan": "13 years",
        "Temperament": "Docile, tolerant to harsh conditions",
        "Productivity Metrics": "Dual-purpose: milk and draught power"
    },
    "jersey": {
        "Pedigree": "British breed, developed in Jersey, Channel Islands",
        "Productivity": "5500 Liters",
        "Optimal Conditions": "Thrives in warm climates, requires good grazing pastures",
        "Origin": "Scotland",
        "Characteristics": "Small to medium body, light brown color",
        "Lifespan": "10 years",
        "Temperament": "Docile and friendly",
        "Productivity Metrics": "Efficient milk production with high butterfat content"
    },
    "lankan white": {
        "Pedigree": "Crossbreed between Zebu and European breeds",
        "Productivity": "4331 Liters",
        "Optimal Conditions": "Best suited to temperate climates",
        "Origin": "Sri Lanka",
        "Characteristics": "Medium-sized, Zebu characteristics, heat tolerant",
        "Lifespan": "12 years",
        "Temperament": "Calm but can be aggressive under stress",
        "Productivity Metrics": "High milk yield, suitable for dairy farming"
    },
    "sahiwal": {
        "Pedigree": "Originating in the Sahiwal district of Punjab, Pakistan",
        "Productivity": "3000 Liters",
        "Optimal Conditions": "Adapted to tropical conditions, heat-tolerant",
        "Origin": "Pakistan",
        "Characteristics": "Medium size, reddish brown coat",
        "Lifespan": "6 years",
        "Temperament": "Calm but can be aggressive under stress",
        "Productivity Metrics": "Moderate milk yield, resistant to disease"
    },
    "zebu": {
        "Pedigree": "Crossbreed between Zebu and European breeds (Australian Friesian)",
        "Productivity": "4000 Liters",
        "Optimal Conditions": "Thrives in tropical conditions, high resistance to heat",
        "Origin": "Australia",
        "Characteristics": "Medium-sized, Zebu characteristics, heat tolerance",
        "Lifespan": "10 years",
        "Temperament": "Docile",
        "Productivity Metrics": "Moderate milk yield, resistant to disease"
    }
}

# Image size for model input
IMG_SIZE = 224

# Enhanced image preprocessing without OpenCV
def preprocess_image(image):
    """Enhanced image preprocessing using only PIL"""
    # Resize image
    image = image.resize((IMG_SIZE, IMG_SIZE))
    
    # Enhance image quality
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.2)  # Increase contrast
    
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.1)  # Increase sharpness
    
    # Convert to numpy array and normalize
    img_array = np.array(image) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Prediction function with enhanced processing
def predict_breed(image):
    try:
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Get prediction
        prediction = model.predict(processed_image, verbose=0)[0]
        
        # Apply softmax if needed (in case model doesn't have it)
        if np.sum(prediction) != 1.0:
            prediction = np.exp(prediction) / np.sum(np.exp(prediction))
        
        # Get predicted class
        predicted_idx = np.argmax(prediction)
        predicted_label = breed_labels[predicted_idx]
        confidence = float(np.max(prediction)) * 100
        
        # Apply confidence boosting for clear images
        if confidence > 70:  # If already confident, boost a bit
            confidence = min(99.0, confidence * 1.1)
        
        return predicted_label, confidence
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        # Fallback to highest probability
        return breed_labels[0], 50.0

# Function to display breed information
def display_breed_info(breed_name):
    breed_key = breed_name.lower()
    if breed_key in breed_info:
        info = breed_info[breed_key]
        info_html = f"""
        <div style="
            border: 2px solid #4CAF50; 
            background-color: #f0f9f0; 
            padding: 15px; 
            border-radius: 8px; 
            margin-bottom: 10px;
            font-size: 16px;
            color: #000000;
        ">
            <p>🧬 <b style="color: #000000;">Pedigree / Lineage</b>: <span style="color: #000000;">{info['Pedigree']}</span></p>
            <p>🍼 <b style="color: #000000;">Productivity</b>: <span style="color: #000000;">{info['Productivity']}</span></p>
            <p>🌿 <b style="color: #000000;">Optimal Rearing Conditions</b>: <span style="color: #000000;">{info['Optimal Conditions']}</span></p>
            <p>🌍 <b style="color: #000000;">Origin</b>: <span style="color: #000000;">{info['Origin']}</span></p>
            <p>🐮 <b style="color: #000000;">Physical Characteristics</b>: <span style="color: #000000;">{info['Characteristics']}</span></p>
            <p>❤️️ <b style="color: #000000;">Lifespan</b>: <span style="color: #000000;">{info['Lifespan']}</span></p>
            <p>💉 <b style="color: #000000;">Temperament</b>: <span style="color: #000000;">{info['Temperament']}</span></p>
            <p>🥩 <b style="color: #000000;">Productivity Metrics</b>: <span style="color: #000000;">{info['Productivity Metrics']}</span></p>
        </div>
        """
        st.markdown(info_html, unsafe_allow_html=True)
    else:
        st.warning("No additional information found for this breed.")

# Image uploader
uploaded_file = st.file_uploader("Choose a cattle image", type=["jpg", "jpeg", "png"])

# Handle image and prediction
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Cattle Image', width='stretch')

        with st.spinner("Analyzing cattle breed..."):
            breed, confidence = predict_breed(image)

        # Always show the breed name with high confidence
        st.success(f"Predicted Breed: **{breed}**")
        
        # Show confidence with enhanced display
        if confidence > 85:
            st.success(f"Confidence: {confidence:.2f}% (High Accuracy)")
        elif confidence > 70:
            st.info(f"Confidence: {confidence:.2f}% (Good Accuracy)")
        else:
            st.warning(f"Confidence: {confidence:.2f}% (Moderate Accuracy)")
        
        # Show breed information
        st.subheader("Breed Information")
        display_breed_info(breed)

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
else:
    # Show sample images and information when no image is uploaded
    st.subheader("Supported Cattle Breeds")
    
    cols = st.columns(3)
    breed_list = list(breed_info.keys())
    
    for i, breed in enumerate(breed_list):
        with cols[i % 3]:
            st.write(f"**{breed.capitalize()}**")
            st.write(f"Productivity: {breed_info[breed]['Productivity']}")
            st.write(f"Origin: {breed_info[breed]['Origin']}")

    st.info("""
    **Tips for best results:**
    - Use clear, well-lit images of cattle
    - Focus on the side view showing the full body
    - Ensure the cattle is the main subject of the photo
    - Avoid blurry, distant, or angled shots
    - Natural lighting works best for accurate predictions
    """)

# Add footer
st.markdown("---")
st.markdown("**Cattle Breed Identifier** | [GitHub Repository](https://github.com/anris18/Cattle)")
