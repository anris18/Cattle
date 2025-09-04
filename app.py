import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io

# Set up the page
st.set_page_config(
    page_title="Cow Breed Identifier",
    page_icon="🐄",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #556B2F;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #F5F5DC;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #8B4513;
        margin-top: 20px;
    }
    .breed-name {
        font-size: 2rem;
        color: #8B4513;
        font-weight: bold;
    }
    .info-box {
        background-color: #F0FFF0;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .stButton>button {
        background-color: #2E8B57;
        color: white;
        font-weight: bold;
    }
    .footer {
        text-align: center;
        margin-top: 30px;
        color: #696969;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.markdown('<h1 class="main-header">🐄 Cow Breed Identifier</h1>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
    This application helps identify cattle breeds from images. Upload a clear photo of a cow, 
    and our AI model will analyze its characteristics to determine the breed.
</div>
""", unsafe_allow_html=True)

# Create two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="sub-header">Upload Cow Image</div>', unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Simulate image processing
        with st.spinner('Analyzing image...'):
            # Simulate processing time
            import time
            time.sleep(2)
            
            # Create a mock "enhanced" image
            enhanced_image = image.copy()
            # In a real app, you would apply actual image enhancement here
            
            # Display enhanced image
            st.image(enhanced_image, caption="Enhanced for Analysis", use_container_width=True)

with col2:
    st.markdown('<div class="sub-header">Breed Identification</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Simulate breed prediction (in a real app, this would come from a model)
        # For demonstration, let's use a random selection from common breeds
        common_breeds = [
            "Holstein Friesian", "Jersey", "Hereford", "Angus", "Brahman",
            "Limousin", "Simmental", "Charolais", "Highland", "Texas Longhorn"
        ]
        
        # Create a more realistic prediction with confidence
        predicted_breed = np.random.choice(common_breeds)
        confidence = round(np.random.uniform(0.85, 0.98), 2)
        
        # Display prediction
        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        st.markdown('<p class="breed-name">' + predicted_breed + '</p>', unsafe_allow_html=True)
        st.metric("Confidence", f"{confidence*100}%")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Breed information
        st.markdown('<div class="sub-header">Breed Characteristics</div>', unsafe_allow_html=True)
        
        # Sample breed info (in a real app, this would come from a database)
        breed_info = {
            "Holstein Friesian": "The most common dairy breed, known for high milk production and distinctive black and white markings.",
            "Jersey": "A smaller dairy breed known for high butterfat content in milk and light brown coloration.",
            "Hereford": "A hardy beef breed known for its red body and white face, excellent foraging ability.",
            "Angus": "A popular beef breed, black in color, known for high-quality marbled meat.",
            "Brahman": "A heat-tolerant beef breed characterized by a large hump and loose skin, originally from India.",
            "Limousin": "A French beef breed known for muscular build and golden-red coloring.",
            "Simmental": "A dual-purpose breed originating from Switzerland, known for rapid growth and good milk production.",
            "Charolais": "A large French beef breed with white coloring and excellent muscling.",
            "Highland": "A Scottish breed with long horns and shaggy coat, well-suited to harsh climates.",
            "Texas Longhorn": "Known for its extremely long horns, lean beef, and historical significance in America."
        }
        
        st.info(breed_info.get(predicted_breed, "Breed information not available."))
        
        # Additional actions
        st.markdown("---")
        st.markdown("**Not the correct breed?**")
        st.button("Try Again with Different Image")
        st.button("Provide Feedback to Improve Model")
    else:
        # Placeholder before image upload
        st.info("Please upload an image of a cow to identify its breed. The results will appear here.")

# Footer
st.markdown("---")
st.markdown('<div class="footer">Cow Breed Identifier v1.0 | AI-Powered Cattle Recognition</div>', unsafe_allow_html=True)
