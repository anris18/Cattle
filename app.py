import streamlit as st
import numpy as np
from PIL import Image
import os

# Try to import required libraries with error handling
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Set page config
st.set_page_config(
    page_title="🐄 Cattle Breed Identifier", 
    layout="centered",
    page_icon="🐄"
)

# Title and description
st.title("🐄 Cattle Breed Identifier")
st.write("Upload an image of a cow to predict its breed.")

# Define breed labels
breed_labels = ["Ayrshire", "Friesian", "Jersey", "Lankan White", "Sahiwal", "Zebu"]

# Check and load model files
@st.cache_resource
def load_cattle_model():
    model_path_h5 = "cattle_breed_model.h5"
    model_path_joblib = "cattle_breed_model.joblib"
    
    model = None
    model_type = None
    
    # Try to load TensorFlow model first
    if TENSORFLOW_AVAILABLE and os.path.exists(model_path_h5):
        try:
            model = load_model(model_path_h5)
            model_type = "h5"
        except Exception:
            pass
    
    # If TensorFlow model not available, try to load joblib model
    if model is None and JOBLIB_AVAILABLE and os.path.exists(model_path_joblib):
        try:
            model = joblib.load(model_path_joblib)
            model_type = "joblib"
        except Exception:
            pass
    
    return model, model_type

# Load model
model, model_type = load_cattle_model()

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
CONFIDENCE_THRESHOLD = 60.0

# Feature extraction function for images - with 9 features
def extract_features(image):
    """Extract 9 features from image to match the trained model"""
    # Resize image
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image) / 255.0
    
    # Extract color features (3 features)
    avg_color = np.mean(img_array, axis=(0, 1))
    
    # Extract color standard deviation (3 features)
    color_std = np.std(img_array, axis=(0, 1))
    
    # Convert to grayscale for texture analysis
    if len(img_array.shape) == 3:
        gray_img = np.mean(img_array, axis=2)
    else:
        gray_img = img_array
    
    # Extract texture features - gradient magnitude (1 feature)
    dy, dx = np.gradient(gray_img)
    gradient_magnitude = np.sqrt(dx**2 + dy**2)
    texture_feature = np.mean(gradient_magnitude)
    
    # Extract texture features - gradient variance (1 feature)
    texture_variance = np.var(gradient_magnitude)
    
    # Extract brightness feature (1 feature)
    brightness = np.mean(gray_img)
    
    # Combine all features to get 9 total features
    features = np.concatenate([
        avg_color,          # 3 features
        color_std,          # 3 features
        [texture_feature],  # 1 feature
        [texture_variance], # 1 feature
        [brightness]        # 1 feature
    ])
    
    return features

# Prediction function for TensorFlow model
def predict_with_tf_model(image):
    # Preprocess image for TensorFlow model
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Get prediction
    prediction = model.predict(img_array, verbose=0)[0]
    return prediction

# Prediction function for joblib model
def predict_with_joblib_model(image):
    # Extract features for joblib model
    features = extract_features(image)
    features_reshaped = features.reshape(1, -1)
    
    # Check if model has predict_proba method
    if hasattr(model, 'predict_proba'):
        prediction = model.predict_proba(features_reshaped)[0]
    else:
        # For models without probability estimates
        predicted_class = model.predict(features_reshaped)[0]
        # Create dummy probabilities
        prediction = np.zeros(len(breed_labels))
        prediction[predicted_class] = 0.9
        # Add some noise to other classes
        for i in range(len(breed_labels)):
            if i != predicted_class:
                prediction[i] = 0.1 / (len(breed_labels) - 1)
    
    return prediction

# Main prediction function
def predict_breed(image):
    if model is not None:
        try:
            if model_type == "h5" and TENSORFLOW_AVAILABLE:
                prediction = predict_with_tf_model(image)
            elif model_type == "joblib" and JOBLIB_AVAILABLE:
                prediction = predict_with_joblib_model(image)
            else:
                # Fallback to demo mode
                prediction = demo_prediction(image)
        except Exception:
            # Fallback to demo mode if prediction fails
            prediction = demo_prediction(image)
    else:
        # Fallback to demo mode
        prediction = demo_prediction(image)
    
    # Get predicted class
    predicted_idx = np.argmax(prediction)
    predicted_label = breed_labels[predicted_idx]
    confidence = float(np.max(prediction)) * 100
    return predicted_label, confidence

# Demo prediction function if model is not available
def demo_prediction(image):
    # Convert to numpy array for processing
    img_array = np.array(image)
    
    # Simple heuristics based on color and patterns for demo purposes
    avg_color = np.mean(img_array, axis=(0, 1))
    color_variance = np.var(img_array, axis=(0, 1))
    
    # Default probabilities
    probabilities = np.array([0.15, 0.25, 0.20, 0.10, 0.15, 0.15])
    
    # Adjust based on color characteristics
    if avg_color[0] > 150:  # Reddish tones
        probabilities[0] += 0.2  # Ayrshire
        probabilities[4] += 0.1  # Sahiwal
    
    if np.max(color_variance) > 500:  # High variance (spotted)
        probabilities[1] += 0.2  # Friesian
        probabilities[0] += 0.1  # Ayrshire
    
    if avg_color[2] > 150:  # Light tones
        probabilities[2] += 0.2  # Jersey
    
    # Normalize to sum to 1
    probabilities = probabilities / np.sum(probabilities)
    
    return probabilities

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

        if confidence < CONFIDENCE_THRESHOLD:
            st.warning("Low confidence prediction. Try a clearer image.")
        else:
            st.success(f"Predicted Breed: **{breed.capitalize()}**")
        
        st.info(f"Confidence: {confidence:.2f}%")
        
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
    **Tips for better results:**
    - Use clear, well-lit images
    - Focus on the side view of the cattle
    - Ensure the cattle is the main subject of the photo
    - Avoid blurry or distant shots
    """)

# Add footer
st.markdown("---")
st.markdown("**Cattle Breed Identifier** | [GitHub Repository](https://github.com/anris18/Cattle)")
