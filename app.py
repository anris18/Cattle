import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
import io

# Create a simple model directly in the code
@st.cache_resource
def create_model():
    # Create a simple CNN model architecture
    model = models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(6, activation='softmax')  # 6 breeds
    ])
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Load or create model
try:
    model = create_model()
    # Initialize with random weights (for demonstration)
    # In a real app, you would load pre-trained weights here
    st.success("✅ Model loaded successfully (demo mode)")
except Exception as e:
    st.error(f"❌ Error creating model: {str(e)}")
    model = None

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

# Breed labels
breed_labels = ["Ayrshire", "Friesian", "Jersey", "Lankan White", "Sahiwal", "Zebu"]
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 60.0

# Streamlit UI
st.set_page_config(page_title="🐄 Cattle Breed Identifier", layout="centered")
st.title("🐄 Cattle Breed Identifier")
st.write("Upload an image of a cow to predict its breed.")

if model is None:
    st.info("🔧 Running in demonstration mode without model support.")

# Image uploader
uploaded_file = st.file_uploader("Choose a cattle image", type=["jpg", "jpeg", "png"])

# Enhanced prediction function with feature extraction
def predict_breed(image):
    # Preprocess image
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image) / 255.0
    
    # Simple feature-based "prediction" (since we don't have trained weights)
    # This is a heuristic approach for demonstration
    img_array = np.expand_dims(img_array, axis=0)
    
    if model is not None:
        try:
            # Get model prediction (will be random without training)
            prediction = model.predict(img_array, verbose=0)[0]
        except:
            # Fallback to heuristic approach if model fails
            prediction = heuristic_prediction(image)
    else:
        # Use heuristic approach if no model
        prediction = heuristic_prediction(image)
    
    predicted_label = breed_labels[np.argmax(prediction)]
    confidence = float(np.max(prediction)) * 100
    return predicted_label, confidence

# Heuristic approach for breed prediction based on image characteristics
def heuristic_prediction(image):
    # Convert to numpy array for processing
    img_array = np.array(image)
    
    # Simple heuristics based on color and patterns
    # This is just for demonstration purposes
    
    # Calculate average color
    avg_color = np.mean(img_array, axis=(0, 1))
    
    # Calculate color variance (for spotting)
    color_variance = np.var(img_array, axis=(0, 1))
    
    # Simple rules based on color characteristics
    # These are arbitrary rules for demonstration only
    
    # Default probabilities
    probabilities = np.array([0.15, 0.25, 0.20, 0.10, 0.15, 0.15])
    
    # Adjust based on color characteristics (very simple heuristics)
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
        ">
            <p>🧬 <b>Pedigree / Lineage</b>: {info['Pedigree']}</p>
            <p>🍼 <b>Productivity</b>: {info['Productivity']}</p>
            <p>🌿 <b>Optimal Rearing Conditions</b>: {info['Optimal Conditions']}</p>
            <p>🌍 <b>Origin</b>: {info['Origin']}</p>
            <p>🐮 <b>Physical Characteristics</b>: {info['Characteristics']}</p>
            <p>❤️️ <b>Lifespan</b>: {info['Lifespan']}</p>
            <p>💉 <b>Temperament</b>: {info['Temperament']}</p>
            <p>🥩 <b>Productivity Metrics</b>: {info['Productivity Metrics']}</p>
        </div>
        """
        st.markdown(info_html, unsafe_allow_html=True)
    else:
        st.warning("⚠️ No additional information found for this breed.")

# Handle image and prediction
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='📷 Uploaded Cattle Image', use_container_width=True)
        
        # Show image analysis
        with st.expander("🔍 Image Analysis"):
            st.write("**Image Details:**")
            st.write(f"- Dimensions: {image.size[0]} x {image.size[1]} pixels")
            st.write(f"- Mode: {image.mode}")
            
            # Simple color analysis
            img_array = np.array(image)
            avg_color = np.mean(img_array, axis=(0, 1))
            st.write(f"- Average RGB color: ({avg_color[0]:.1f}, {avg_color[1]:.1f}, {avg_color[2]:.1f})")

        with st.spinner("🔍 Analyzing cattle breed..."):
            breed, confidence = predict_breed(image)

        if confidence < CONFIDENCE_THRESHOLD:
            st.warning("⚠️ Low confidence prediction. This might not be accurate.")
        else:
            st.success(f"✅ Predicted Breed: **{breed}**")
        
        st.info(f"🔎 Confidence: {confidence:.2f}%")
        
        # Show breed information
        st.subheader("📚 Breed Information")
        display_breed_info(breed)
        
        # Disclaimer
        st.info("""
        **ℹ️ Note:** This is a demonstration application. 
        For accurate breed identification, a properly trained model with extensive dataset is required.
        """)

    except Exception as e:
        st.error(f"⚠️ Error processing image: {str(e)}")
else:
    # Show sample images and information when no image is uploaded
    st.subheader("📋 Supported Cattle Breeds")
    
    cols = st.columns(3)
    breed_list = list(breed_info.keys())
    
    for i, breed in enumerate(breed_list):
        with cols[i % 3]:
            st.write(f"**{breed.capitalize()}**")
            st.write(f"Productivity: {breed_info[breed]['Productivity']}")
            st.write(f"Origin: {breed_info[breed]['Origin']}")

    st.info("""
    **📸 Tips for better results:**
    - Use clear, well-lit images
    - Focus on the side view of the cattle
    - Ensure the cattle is the main subject of the photo
    - Avoid blurry or distant shots
    """)
