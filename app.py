import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import timm
import os
import base64
import requests
from io import BytesIO
import json
import time
import random
import csv
import pandas as pd
from datetime import datetime
from gtts import gTTS
from io import BytesIO

# Set page config first (should be the first Streamlit command)
st.set_page_config(page_title="🐄 Cattle Breed Identifier", layout="centered", initial_sidebar_state="collapsed")

# Translation functionality
def get_translation(key, language="en"):
    translations = {
        "en": {
            "title": "🐄 Indian Cattle Breed Identifier",
            "subtitle": "Discover the rich diversity of Indian bovine breeds",
            "upload_info": "📁 Upload an image of a cow or buffalo to identify its breed",
            "marketplace_button": "🐂 Cattle Marketplace",
            "upload_label": "Choose a cattle image",
            "drag_drop": "Drag and drop an image here",
            "file_limit": "Limit 200MB per file • JPG, JPEG, PNG",
            "browse_files": "or browse files",
            "help_text": "Select an image of an Indian cattle breed",
            "image_caption": "📷 Uploaded Cattle Image",
            "analyzing": "🔍 Analyzing breed characteristics...",
            "predicted_breed": "✅ Predicted Breed:",
            "confidence": "🔎 Confidence:",
            "breed_info": "📚 Breed Information",
            "pedigree": "Pedigree / Lineage",
            "productivity": "Productivity",
            "rearing_conditions": "Optimal Rearing Conditions",
            "origin": "Origin",
            "physical_chars": "Physical Characteristics",
            "lifespan": "Lifespan (Years)",
            "temperament": "Temperament",
            "productivity_metrics": "Productivity Metrics",
            "physical_measurements": "📏 Physical Measurements",
            "body_length": "Body Length",
            "height_withers": "Height at Withers",
            "chest_width": "Chest Width",
            "rump_angle": "Rump Angle",
            "refresh": "🔄 Refresh the page to analyze another image",
            "heritage": "🐄 Celebrating India's rich bovine heritage",
            "marketplace_title": "🐂 Cattle Marketplace",
            "marketplace_subtitle": "Buy and Sell Quality Cattle",
            "back_button": "← Back to Breed Identifier",
            "marketplace_info": "Browse available cattle listings below. Contact sellers directly for purchases.",
            "price": "Price",
            "age": "Age",
            "milk_yield": "Milk Yield",
            "lactation_stage": "Lactation Stage",
            "vaccination": "Vaccination",
            "seller": "Seller",
            "contact": "Contact",
            "location": "Location",
            "add_listing": "Add Your Listing",
            "cattle_breed": "Cattle Breed",
            "submit_listing": "Submit Listing",
            "listing_submitted": "Listing submitted! It will appear below.",
            "description": "Description (optional)",
            "chat_title": "Cattle Assistant",
            "prediction_error": "❌ Prediction error",
            "processing_error": "⚠ Error processing image",
            "confidence_error": "Could not confidently identify the breed. Try a clearer image with the animal facing sideways.",
            "incomplete_info": "⚠ Incomplete breed info.",
            "info_parsing_error": "❌ Error parsing breed info:",
            "no_info": "⚠ No additional information found for this breed."
        },
        "hi": {
            "title": "🐄 भारतीय मवेशी नस्ल पहचानकर्ता",
            "subtitle": "भारतीय बोवाइन नस्लों की समृद्ध विविधता की खोज करें",
            "upload_info": "📁 अपनी नस्ल की पहचान करने के लिए गाय या भैंस की एक छवि अपलोड करें",
            "marketplace_button": "🐂 मवेशी बाजार",
            "upload_label": "एक मवेशी छवि चुनें",
            "drag_drop": "यहां एक छवि खींचें और छोड़ें",
            "file_limit": "प्रति फ़ाइल 200MB की सीमा • JPG, JPEG, PNG",
            "browse_files": "या फ़ाइलें ब्राउज़ करें",
            "help_text": "एक भारतीय मवेशी नस्ल की छवि का चयन करें",
            "image_caption": "📷 अपलोड की गई मवेशी छवि",
            "analyzing": "🔍 नस्ल की विशेषताओं का विश्लेषण किया जा रहा है...",
            "predicted_breed": "✅ अनुमानित नस्ल:",
            "confidence": "🔎 आत्मविश्वास:",
            "breed_info": "📚 नस्ल की जानकारी",
            "pedigree": "वंशावली / वंश",
            "productivity": "उत्पादकता",
            "rearing_conditions": "इष्टतम पालन की स्थिति",
            "origin": "मूल",
            "physical_chars": "शारीरिक विशेषताएं",
            "lifespan": "जीवनकाल (वर्ष)",
            "temperament": "स्वभाव",
            "productivity_metrics": "उत्पादकता मेट्रिक्स",
            "physical_measurements": "📏 शारीरिक माप",
            "body_length": "शरीर की लंबाई",
            "height_withers": "कंधे की ऊंचाई",
            "chest_width": "छाती की चौड़ाई",
            "rump_angle": "रंप कोण",
            "refresh": "🔄 किसी अन्य छवि का विश्लेषण करने के लिए पृष्ठ ताज़ा करें",
            "heritage": "🐄 भारत की समृद्ध बोवाइन विरासत का जश्न",
            "marketplace_title": "🐂 मवेशी बाजार",
            "marketplace_subtitle": "गुणवत्तापूर्ण मवेशी खरीदें और बेचें",
            "back_button": "← ब्रीड आइडेंटिफायर पर वापस जाएं",
            "marketplace_info": "नीचे उपलब्ध मवेशी सूचियां देखें। खरीदारी के लिए सीधे विक्रेताओं से संपर्क करें।",
            "price": "कीमत",
            "age": "उम्र",
            "milk_yield": "दूध उत्पादन",
            "lactation_stage": "दुग्धावस्था",
            "vaccination": "टीकाकरण",
            "seller": "विक्रेता",
            "contact": "संपर्क",
            "location": "स्थान",
            "add_listing": "अपनी लिस्टिंग जोड़ें",
            "cattle_breed": "मवेशी नस्ल",
            "submit_listing": "लिस्टिंग सबमिट करें",
            "listing_submitted": "लिस्टिंग सबमिट की गई! यह नीचे दिखाई देगी।",
            "description": "विवरण (वैकल्पिक)",
            "chat_title": "मवेशी सहायक",
            "prediction_error": "❌ भविष्यवाणी त्रुटि",
            "processing_error": "⚠ छवि प्रसंस्करण में त्रुटि",
            "confidence_error": "नस्ल को विश्वास के साथ पहचान नहीं सका। जानवर को बगल में दिखाने वाली स्पष्ट छवि आज़माएं।",
            "incomplete_info": "⚠ अधूरी नस्ल जानकारी।",
            "info_parsing_error": "❌ नस्ल जानकारी पार्स करने में त्रुटि:",
            "no_info": "⚠ इस नस्ल के लिए कोई अतिरिक्त जानकारी नहीं मिली।"
        },
        "te": {
            "title": "🐄 భారతీయ పశువుల జాతి గుర్తింపు",
            "subtitle": "భారతీయ పశువుల జాతుల సంపన్న వైవిధ్యాన్ని కనుగొనండి",
            "upload_info": "📁 దాని జాతిని గుర్తించడానికి ఒక ఆవు లేదా ఎదురు చిత్రాన్ని అప్లోడ్ చేయండి",
            "marketplace_button": "🐂 పశువుల మార్కెట్",
            "upload_label": "ఒక పశు చిత్రాన్ని ఎంచుకోండి",
            "drag_drop": "ఇక్కడ ఒక చిత్రాన్ని లాగండి మరియు వదలండి",
            "file_limit": "ఫైల్ కు 200MB పరిమితి • JPG, JPEG, PNG",
            "browse_files": "లేదా ఫైల్లను బ్రౌజ్ చేయండి",
            "help_text": "ఒక భారతీయ పశు జాతి చిత్రాన్ని ఎంచుకోండి",
            "image_caption": "📷 అప్లోడ్ చేసిన పశు చిత్రం",
            "analyzing": "🔍 జాతి లక్షణాలను విశ్లేషిస్తోంది...",
            "predicted_breed": "✅ అంచనా వేసిన జాతి:",
            "confidence": "🔎 నమ్మకం:",
            "breed_info": "📚 జాతి సమాచారం",
            "pedigree": "వంశం / వంశావళి",
            "productivity": "ఉత్పాదకత",
            "rearing_conditions": "ఆదర్శ పెంపకడ పరిస్థితులు",
            "origin": "మూలం",
            "physical_chars": "భౌతిక లక్షణాలు",
            "lifespan": "ఆయుష్ (సంవత్సరాలు)",
            "temperament": "స్వభావం",
            "productivity_metrics": "ఉత్పాదకత మెట్రిక్స్",
            "physical_measurements": "📏 భౌతిక కొలతలు",
            "body_length": "శరీర పొడవు",
            "height_withers": "భుజాల ఎత్తు",
            "chest_width": "ఛాతీ వెడల్పు",
            "rump_angle": "రంప్ కోణం",
            "refresh": "🔄 మరొక చిత్రాన్ని విశ్లేషించడానికి పేజీని రిఫ్రెష్ చేయండి",
            "heritage": "🐄 భారతదేశం యొక్క సంపన్న పశు వారసత్వాన్ని జరుపుకుంటోంది",
            "marketplace_title": "🐂 పశువుల మార్కెట్",
            "marketplace_subtitle": "నాణ్యత గల పశువులను కొనండి మరియు విక్రయించండి",
            "back_button": "← బ్రీడ్ ఐడెంటిఫైయర్‌కు తిరిగి వెళ్లండి",
            "marketplace_info": "క్రింద అందుబాటులో ఉన్న పశువుల జాబితాలను బ్రౌజ్ చేయండి. కొనుగోలు కోసం నేరుగా విక్రేతలను సంప్రదించండి.",
            "price": "ధర",
            "age": "వయస్సు",
            "milk_yield": "పాలు దిగుబడి",
            "lactation_stage": "పాల ఉత్పత్తి దశ",
            "vaccination": "తడిపించడం",
            "seller": "విక్రేత",
            "contact": "సంప్రదింపు",
            "location": "స్థానం",
            "add_listing": "మీ లిస్టింగ్‌ని జోడించండి",
            "cattle_breed": "పశు జాతి",
            "submit_listing": "లిస్టింగ్ సమర్పించండి",
            "listing_submitted": "లిస్టింగ్ సమర్పించబడింది! ఇది క్రింద కనిపిస్తుంది.",
            "description": "వివరణ (ఐచ్ఛికం)",
            "chat_title": "పశు సహాయక",
            "prediction_error": "❌ అంచనా దోషం",
            "processing_error": "⚠ చిత్ర ప్రాసెసింగ్ లో దోషం",
            "confidence_error": "జాతిని నమ్మకంగా గుర్తించలేకపోయింది. జంతువును పక్కన చూపించే స్పష్టమైన చిత్రాన్ని ప్రయత్నించండి.",
            "incomplete_info": "⚠ అసంపూర్ణ జాతి సమాచారం.",
            "info_parsing_error": "❌ జాతి సమాచారాన్ని పార్స్ చేయడంలో దోషం:",
            "no_info": "⚠ ఈ జాతి కోసం అదనపు సమాచారం లేదు."
        }
    }
    
    return translations.get(language, translations["en"]).get(key, key)

def language_selector():
    st.sidebar.markdown("---")
    st.sidebar.header("🌐 Language / भाषा / భाष")
    language = st.sidebar.radio("Select Language", ["English", "Hindi", "Telugu"], index=0, label_visibility="collapsed")
    
    lang_map = {
        "English": "en",
        "Hindi": "hi", 
        "Telugu": "te"
    }
    
    return lang_map[language]

def translate_breed_info(info_text, language="en"):
    # Simple translation mapping for breed information
    translations = {
        "en": {
            "ORIGINATED IN": "ORIGINATED IN",
            "NA (Draft breed)": "NA (Draft breed)",
            "ADAPTED TO": "ADAPTED TO",
            "INDIA": "INDIA",
            "MEDIUM": "MEDIUM",
            "LARGE": "LARGE",
            "SMALL": "SMALL",
            "HARDY": "HARDY",
            "DOCILE": "DOCILE",
            "ACTIVE": "ACTIVE",
            "Liters": "Liters",
            "PRIMARILY USED FOR": "PRIMARILY USED FOR"
        },
        "hi": {
            "ORIGINATED IN": "की उत्पत्ति",
            "NA (Draft breed)": "NA (ड्राफ्ट नस्ल)",
            "ADAPTED TO": "के लिए अनुकूलित",
            "INDIA": "भारत",
            "MEDIUM": "मध्यम",
            "LARGE": "बड़ा",
            "SMALL": "छोटा",
            "HARDY": "हार्डी",
            "DOCILE": "डोसाइल",
            "ACTIVE": "सक्रिय",
            "Liters": "लीटर",
            "PRIMARILY USED FOR": "मुख्य रूप से के लिए उपयोग किया जाता है"
        },
        "te": {
            "ORIGINATED IN": "వద్ద ఉద్భవించింది",
            "NA (Draft breed)": "NA (డ్రాఫ్ట్ జాతి)",
            "ADAPTED TO": "కు అనుకూలీకరించబడింది",
            "INDIA": "భారతదేశం",
            "MEDIUM": "మధ్యస్థ",
            "LARGE": "పెద్ద",
            "SMALL": "చిన్న",
            "HARDY": "హార్డీ",
            "DOCILE": "డోసైల్",
            "ACTIVE": "క్రియాశీల",
            "Liters": "లీటర్లు",
            "PRIMARILY USED FOR": "ప్రధానంగా ఉపయోగించబడుతుంది"
        }
    }
    
    if language == "en":
        return info_text
    
    # Simple word-by-word translation
    translated_text = info_text
    for eng, trans in translations[language].items():
        translated_text = translated_text.replace(eng, trans)
    
    return translated_text

# Add custom CSS for styling
def set_custom_style():
    st.markdown(
        """
        <style>
        .stApp {
            background: url('https://images.unsplash.com/photo-1527153857715-3908f2bae5e8?ixlib=rb-4.0.3&auto=format&fit=crop&w=2089&q=80');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        
        /* Add overlay to ensure text readability */
        .main .block-container {
            background-color: rgba(255, 255, 255, 0.9);
            border-radius: 15px;
            padding: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .main-header {
            color: #2c3e50;
            text-align: center;
            font-size: 2.8rem;
            font-weight: bold;
            margin-bottom: 1rem;
            font-family: 'Arial', sans-serif;
            text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.8);
        }
        
        .sub-header {
            color: #34495e;
            text-align: center;
            font-size: 1.3rem;
            margin-bottom: 2rem;
            font-family: 'Arial', sans-serif;
            text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.8);
        }
        
        .prediction-box {
            background-color: rgba(255, 255, 255, 极95);
            padding: 20px;
            border-radius: 12px;
            border-left: 5px solid #3498db;
            margin: 15px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            color: #2c3e50;
        }
        
        .breed-info {
            background-color: rgba(248, 249, 250, 0.95);
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #27ae60;
            margin: 15px 0;
            color: #2c3e50;
        }
        
        .physical-measurements {
            background-color: rgba(232, 244, 248, 0.95);
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #e74c3c;
            margin: 15极 0;
            color: #2c3e50;
        }
        
        .upload-box {
            background-color: rgba(255, 255, 255, 0.95);
            padding: 20px;
            border-radius: 10px;
            border: 2px dashed #3498db;
            margin: 15px 0;
            color: #2c3e50;
        }
        
        .info-text {
            background-color: rgba(232, 244, 248, 0.95);
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            color: #2c3e50;
        }
        
        .stButton>button {
            background-color: #3498db;
            color: white;
            border-radius: 8px;
            padding: 10px 24px;
            font-weight: bold;
        }
        
        /* Ensure all text is visible */
        .stMarkdown, .stText, .stCaption, .stSuccess, .极Warning, .stError, .stInfo {
            color: #2c3e50 !important;
        }
        
        /* Prediction text styling */
        .prediction-text {
            color: #498db1;
            font-weight: bold;
            font-size: 1.5rem;
        }
        
        /* Confidence text styling */
        .confidence-text {
            color: #3498db;
            font-weight: bold;
            font-size: 1.2rem;
        }
        
        /* Footer styling */
        .footer {
            text-align: center;
            color: #2c3e50;
            padding: 10px;
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 10px;
            margin-top: 20px;
        }
        
        /* Chatbot styling */
        .chat-icon {
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 1000;
            background-color: #3498db;
            color: white;
            border-radius: 50%;
            width: 60px;
            height: 60px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            font-size: 24px;
        }
        
        .chat-container {
            position: fixed;
            bottom: 90px;
            right: 20px;
            width: 350px;
            height: 450px;
            background-color: white;
            border-radius: 15px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
            z-index: 1000;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        .chat-header {
            background-color: #3498db;
            color: white;
            padding: 15px;
            font-weight: bold;
            border-top-left-radius: 15px;
            border-top-right-radius: 15px;
        }
        
        .chat-messages {
            flex: 1;
            padding: 15px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        
        .message {
            max-width: 80%;
            padding: 10px 15px;
            border-radius: 15px;
            margin-bottom: 10px;
        }
        
        .bot-message {
            background-color: #f1f1f1;
            align-self: flex-start;
            border-bottom-left-radius: 5px;
        }
        
        .user-message {
            background-color: #3498db;
            color: white;
            align-self: flex-end;
            border-bottom-right-radius: 5px;
        }
        
        .chat-input {
            display: flex;
            padding: 10px;
            border-top: 1px solid #ddd;
            background-color: white;
        }
        
        .chat-input input {
            flex: 1;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 20极;
            outline: none;
        }
        
        .chat-input button {
            margin-left: 10px;
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 20px;
            padding: 10px 15px;
            cursor: pointer;
        }
        
        .quick-replies {
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            padding: 10px;
            background-color: #f9f9f9;
        }
        
        .quick-reply {
            background-color: #e8f4f8;
            border: 1px solid #3498db;
            border-radius: 15px;
            padding: 5极 10px;
            font-size: 12px;
            cursor: pointer;
        }
        
        .quick-reply:hover {
            background-color: #3498db;
            color: white;
        }
        
        /* Marketplace styling */
        .cattle-card {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .cattle-name {
            font-weight: bold;
            font-size: 18px;
            color: #2c3e50;
        }
        
        .cattle-price {
            color: #27ae60;
            font-weight: bold;
            font-size: 16px;
        }
        
        .seller-info {
            color: #7f8c8d;
            font-size: 14px;
        }
        
        /* Chat toggle button */
        .chat-toggle {
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 1000;
        }
        
        /* Sidebar styling */
        .sidebar .sidebar-content {
            background-color: rgba(255, 255, 255, 0.95);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_custom_style()

# Language selection
language = language_selector()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Breed labels in model output order
breed_labels = [
    "Alambadi", "Amritmahal", "Ayrshire", "Banni", "Bargur", 
    "Bhadawari", "Brown_Swiss", "Dangi", "Deoni", "Gir", 
    "Guernsey", "Hallikar", "Hariana", "Holstein_Friesian", "Jaffrabadi", 
    "Jersey", "Kangayam", "Kankre极", "Kasargod", "Kenkatha", 
    "Kherigarh", "Khillari", "Krishna_Valley", "Malnad_gidda", "Mehsana", 
    "Murrah", "Nagori", "Nagpuri", "Nili_Ravi", "Nimari", 
    "Ongole", "Pulikulam", "Rathi", "Red_Dane", "Red_Sindhi", 
    "Sahiwal", "Surti", "Tharparkar", "Toda", "Umblachery", 
    "Vechur"
]

# Load model function for PyTorch
@st.cache_resource
def load_model():
    try:
        # Define your model architecture (must match training)
        model = timm.create_model("resnet50", pretrained=False, num_classes=len(breed_labels))
        
        # Load the saved weights
        checkpoint_path = "best_resnet50_indian_bovine_breeds.pth"
        
        if not os.path.exists(checkpoint_path):
            st.error(f"Model file '{checkpoint_path}' not found. Please make sure it's in the same directory.")
            return None
            
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            # Try loading directly (might be just the state dict)
            model.load_state_dict(checkpoint)
            
        model.to(device)
        model.eval()
        return model
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("💡 Make sure your model file is in the correct format and architecture matches.")
        return None

# Load the model
model = load_model()

# Only continue if model loaded successfully
if model is None:
    st.stop()

# Breed information with physical measurements - manually added for all breeds
breed_info_raw = {
    "alambadi": {
        "info": """ORIGINATED IN ALAMBADI VILLAGE OF DHARMAPURI DISTRICT, TAMIL NADU
NA (Draft breed)
ADAPTED TO TROPICAL CLIMATES, HEAT TOLERANT
INDIA (Tamil Nadu)
MEDIUM TO LARGE SIZE, DARK GREY TO BLACK COLOR
15-20
HARDY AND ACTIVE
PRIMARILY USED FOR DRAFT PURPOSES""",
        "measurements": {
            "body_length": "140-150 cm",
            "height_withers": "130-140 cm",
            "chest_width": "45-50 cm",
            "rump_angle": "5-7 degrees"
        }
    },
    "amritmahal": {
        "info": """ORIGINATED IN KARNATAKA, INDIA
NA (Draft breed)
WELL ADAPTED TO TROPICAL CLIMATES
INDIA (Karnataka)
MEDIUM SIZE, GREYISH WHITE TO DARK GREY COLOR
15-20
ACTIVE AND FIERY TEMPERAMENT
PRIMARILY A DRAFT BREED""",
        "measurements": {
            "body_length": "145-155 cm",
            "height_withers": "135-145 cm",
            "chest_width": "48-53 cm",
            "rump_angle": "6-8 degrees"
        }
    },
    "ayrshire": {
        "info": """DEVELOPED IN THE COUNTY OF AYRSHIRE IN SOUTHWESTERN SCOTLAND
4500-6000 Liters
BEST SUITED TO TEMPERATE CLIMATES
SCOTLAND
MEDIUM SIZE, REDDISH-BROWN AND WHITE SPOTS
12-15
ALERT AND ACTIVE
HIGH MILK QUALITY WITH GOOD FAT CONTENT""",
        "measurements": {
            "body_length": "150-160 cm",
            "height_withers": "140-150 cm",
            "chest_width": "50-55 cm",
            "rump_angle": "4-6 degrees"
        }
    },
    "banni": {
        "info": """NATIVE TO KUTCH DISTRICT OF GUJARAT
1800-2500 Liters
ADAPTED TO ARID AND SEMI-ARID REGIONS
INDIA (Gujarat)
MEDIUM SIZE, WHITE TO GREY COLOR
12-15
HARDY AND DOCILE
GOOD MILK YIELD UNDER LOW INPUT CONDITIONS""",
        "measurements": {
            "body_length": "145极155 cm",
            "height_withers": "135-145 cm",
            "chest_width": "48-53 cm",
            "rump_angle": "5-7 degrees"
        }
    },
    "bargur": {
        "info": """ORIGINATED IN BARGUR HILLS OF TAMIL NAD极
NA (Draft breed)
ADAPTED TO HILLY TERRAIN
INDIA (Tamil Nadu)
MEDIUM SIZE, BROWN WITH WHITE MARKINGS
15-20
FIERY TEMPERAMENT, FAST WALKER
PRIMARILY USED FOR DRAFT PURPOSES""",
        "measurements": {
            "body_length": "140-150 cm",
            "height_withers": "130-140 cm",
            "chest_width": "45-50 cm",
            "rump_angle": "6-8 degrees"
        }
    },
    "bhadawari": {
        "info": """NATIVE TO BHADAWAR REGION OF UTTAR PRADESH
800-1200 Liters
ADAPTED TO TROPICAL CLIMATES
INDIA (Uttar Pradesh)
SMALL TO MEDIUM SIZE, LIGHT BROWN TO DARK BROWN
15-20
DOCILE AND HARDY
KNOWN FOR HIGH FAT CONTENT IN MILK""",
        "measurements": {
            "body_length": "135-145 cm",
            "height_withers": "125-135 cm",
            "chest_width": "42-47 cm",
            "rump_angle": "5-7 degrees"
        }
    },
    "brown_swiss": {
        "info": """ORIGINATED IN SWITZERLAND
6000-8000 Liters
ADAPTED TO VARIOUS CLIMATIC CONDITIONS
SWITZERLAND
LARGE SIZE, BROWN TO GREY COLOR
12-15
DOCILE AND CALm
GOOD MILK PRODUCTION WITH HIGH PROTEIN CONTENT""",
        "measurements": {
            "body_length": "155-165 cm",
            "height_withers": "145-155 cm",
            "chest_width": "52-57 cm",
            "rump_angle": "4-6 degrees"
        }
    },
    "dangi": {
        "info": """NATIVE TO DANG DISTRICT OF GUJARAT
NA (Draft breed)
ADAPTED TO HIGH RAINFALL AREA
INDIA (Gujarat)
MEDIUM SIZE, WHITE WITH BLACK OR RED SPOTS
15-20
HARDY AND ACTIVE
PRIMARILY USED FOR DRAFT PURPOSES""",
        "measurements": {
            "body_length": "140-150 cm",
            "height_withers": "130-140 cm",
            "chest_width": "45-50 cm",
            "rump_angle": "6-8 degrees"
        }
    },
    "deoni": {
        "info": """ORIGINATED IN BIDAR DISTRICT OF KARNATAKA
1000-1500 Liters
ADAPTED TO TROPICAL CLIMATES
INDIA (Karnataka)
MEDIUM TO LARGE SIZE, WHITE WITH BLACK MARKINGS
12-15
DOCILE AND HARDY
DUAL-PURPOSE BREED FOR MILK AND DRAFT""",
        "measurements": {
            "body_length": "145-155 cm",
            "height_withers": "135-145 cm",
            "chest_width": "48-53 cm",
            "rump_angle": "5-7 degrees"
        }
    },
    "gir": {
        "info": """ORIGINATED IN GIR FOREST OF GUJARAT
1500-2000 Liters
ADAPTED TO HOT CLIMATES
INDIA (Gujarat)
LARGE SIZE, REDDISH BROWN WITH WHITE SPOTS
12-15
DOCILE AND GENTLE
GOOD MILK YIELD WITH HIGH FAT CONTENT""",
        "measurements": {
            "body_length": "155-165 cm",
            "height_withers": "145-155 cm",
            "chest_width": "52-57 cm",
            "rump_angle": "5-7 degrees"
        }
    },
    "guernsey": {
        "info": """ORIGINATED IN ISLE OF GUERNSEY, UK
5000-6000 Liters
PREFERS TEMPERATE CLIMATES
UNITED KINGDOM
MEDIUM SIZE, FAWN AND WHITE COLOR
10-12
DOCILE AND GENTLE
RICH GOLDEN-COLORED MILK WITH HIGH FAT""",
        "measurements": {
            "body_length": "150-160 cm",
            "height_withers": "140-150 cm",
            "chest_width": "50-55 cm",
            "rump_angle": "4-6 degrees"
        }
    },
    "hallikar": {
        "info": """ORIGINATED IN KARNATAKA, INDIA
NA (Draft breed)
ADAPTED TO TROPICAL CLIMATES
INDIA (Karnataka)
MEDIUM SIZE, GREY TO DARK GREY COLOR
15-20
ACTIVE AND FIERY
PRIMARILY USED AS DRAFT ANIMALS""",
        "measurements": {
            "body_length": "145-155 cm",
            "height_withers": "135-145 cm",
            "chest_width": "48-53极",
            "rump_angle": "6-8 degrees"
        }
    },
    "hariana": {
        "info": """ORIGINATED IN HARYANA AND PUNJAB REGIONS
1000-1500 Liters
ADAPTED TO NORTH INDIAN CLIMATE
INDIA (Haryana, Punjab)
MEDIUM SIZE, WHITE TO LIGHT GREY COLOR
12-15
DOCILE AND HARDY
DUAL-PURPOSE BREED FOR MIL极 AND DRAFT""",
        "measurements": {
            "body_length": "150-160 cm",
            "height_withers": "140-150 cm",
            "chest_width": "50-55 cm",
            "rump_angle": "5极7 degrees"
        }
    },
    "holstein_friesian": {
        "info": """ORIGINATED IN NETHERLANDS AND GERMANY
7000-9000 Liters
ADAPTED TO TEMPERATE CLIMATES
NETHERLANDS/GERMANY
LARGE SIZE, BLACK AND WHITE OR RED AND WHITE
10-12
DOCILE AND CALM
HIGHEST MILK PRODUCING DAIRY BREED""",
        "measurements": {
            "body_length": "160-170 cm",
            "height_withers": "150-160 cm",
            "chest_width": "55-60 cm",
            "rump_angle": "4-6 degrees"
        }
    },
    "jaffrabadi": {
        "info": """NATIVE TO GUJARAT, INDIA
1500-2500 Liters
ADAPTED TO TROPICAL CLIMATES
INDIA (Gujarat)
LARGE SIZE, BLACK WITH WHITE MARKINGS
12-15
HARDY AND STRONG
GOOD BUFFALO BREED FOR MILK PRODUCTION""",
        "measurements": {
            "body_length": "155-165 cm",
            "height_withers": "145-155 cm",
            "chest_width": "52-57 cm",
            "rump_angle": "5-7 degrees"
        }
    },
    "jersey": {
        "info": """ORIGINATED IN JERSEY ISLAND, UK
5000-6000 Liters
ADAPTED TO VARIOUS CLIMATES
UNITED KINGDOM
SMALL TO MEDIUM SIZE, LIGHT BROWN TO DARK BROWN
10-12
DOCILE AND GENTLE
HIGH EFFICIENCY IN MILK PRODUCTION""",
        "measurements": {
            "body_length": "140-150 cm",
            "height_withers": "130-140 cm",
            "chest_width": "45-50 cm",
            "rump_angle": "6-8 degrees"
        }
    },
    "kangayam": {
        "info": """ORIGINATED IN TAMIL NADU, INDIA
NA (Draft breed)
ADAPTED TO TROPICAL CLIMATES
INDIA (Tamil Nadu)
MEDIUM SIZE, GREY TO WHITE COLOR
15-20
HARDY AND ACTIVE
PRIMARILY USED FOR DRAFT PURPOSES""",
        "measurements": {
            "body_length": "145-155 cm",
            "height_withers": "135-145 cm",
            "chest_width": "48-53 cm",
            "rump_angle": "6-8 degrees"
        }
    },
    "kankrej": {
        "info": """ORIGINATED IN GUJARAT, INDIA
NA (Draft breed)
ADAPTED TO ARID CLIMATES
INDIA (Gujarat)
LARGE SIZE, GREY TO SILVERY GREY COLOR
15-20
STRONG AND HARDY
ONE OF THE BEST INDIAN DRAFT BREEDS""",
        "measurements": {
            "body_length": "155-165 cm",
            "height_withers": "145-155 cm",
            "chest_width": "52-57 cm",
            "rump_angle": "5-7 degrees"
        }
    },
    "kasargod": {
        "info": """NATIVE TO KASARAGOD DISTRICT OF KERALA
500-800 Liters
ADAPTED TO HIGH HUMIDITY CONDITIONS
INDIA (Kerala)
SMALL SIZE, REDDISH BROWN TO BLACK COLOR
12-15
HARDY AND DOCILE
SMALL INDIGENOUS CATTLE BREED""",
        "measurements": {
            "body_length": "130-140 cm",
            "height_withers": "120-130 cm",
            "chest_width": "40-45 cm",
            "rump_angle": "6-8 degrees"
        }
    },
    "kenkatha": {
        "info": """ORIGINATED IN BUNDELKHAND REGION
NA (Draft breed)
ADAPTED TO DRY CLIMATES
IND极 (Uttar Pradesh)
SMALL SIZE, GREY TO WHITE COLOR
15-20
HARDY AND STRONG
PRIMARILY USED FOR AGRICULTURAL WORK""",
        "measurements": {
            "body_length": "135-145 cm",
            "height_withers": "125-135 cm",
            "chest_width": "42-47 cm",
            "rump_angle": "6-8 degrees"
        }
    },
    "kherigarh": {
        "info": """NATIVE TO UTTAR PRADESH, INDIA
NA (Draft breed)
ADAPTED TO NORTH INDIAN CLIMATE
INDIA (Uttar Pradesh)
SMALL SIZE, GREY TO WHITE COLOR
15-20
HARDY AND ACTIVE
PRIMARILY USED FOR DRAFT PURPOSES""",
        "measurements": {
            "body_length": "135-145 cm",
            "height_withers": "125-135 cm",
            "chest_width": "42-47 cm",
            "rump_angle": "6-8 degrees"
        }
    },
    "khillari": {
        "info": """ORIGINATED IN MAHARASHTRA, INDIA
NA (Draft breed)
ADAPTED TO DRY CLIMATES
INDIA (Maharashtra)
MEDIUM SIZE, WHITE TO GREY COLOR
15-20
HARDY AND ACTIVE
PRIMARILY A DRAFT BREED""",
        "measurements": {
            "body_length": "145-155 cm",
            "height_withers": "135-145 cm",
            "chest_width": "48-53 cm",
            "rump_angle": "6-8 degrees"
        }
    },
    "krishna_valley": {
        "info": """ORIGINATED IN KRISHNA VALLEY OF MAHARASHTRA
NA (Draft breed)
ADAPTED TO TROPICAL CLIMATES
INDIA (Maharashtra)
LARGE SIZE, WHITE WITH BLACK MARKINGS
15-20
GENTLE AND DOCILE
GOOD DRAFT BREED WITH HEAVY BODY""",
        "measurements": {
            "body_length": "155-165 cm",
            "height_withers": "极45-155 cm",
            "chest_width": "52-57 cm",
            "rump_angle": "5-7 degrees"
        }
    },
    "malnad_gidda": {
        "info": """NATIVE TO WESTERN GHATS OF KARNATAKA
500-800 Liters
ADAPTED TO HILLY TERRAIN
INDIA (Karnataka)
SMALL SIZE, BLACK OR BROWN COLOR
12-15
HARDY AND DOCILE
SMALL INDIGENOUS CATTLE BREED""",
        "measurements": {
            "body_length": "130-140 cm",
            "height_withers": "120-130 cm",
            "chest_width": "40-45 cm",
            "rump_angle": "6-8 degrees"
        }
    },
    "mehsana": {
        "info": """ORIGINATED IN GUJARAT, INDIA
1500-2000 Liters
ADAPTED TO TROPICAL CLIMATES
INDIA (Gujarat)
MEDIUM SIZE, BLACK WITH WHITE MARKINGS
12-15
DOCILE AND HARDY
GOOD BUFFALO BREED FOR MILK PRODUCTION""",
        "measurements": {
            "body_length": "150-160 cm",
            "height_withers": "140-150 cm",
            "chest_width": "50-55 cm",
            "rump_angle": "5-7 degrees"
        }
    },
    "murrah": {
        "info": """ORIGINATED IN HARYANA, INDIA
1800-2500 Liters
ADAPTED TO NORTH INDIAN CLIMATE
INDIA (Haryana)
MEDIUM SIZE, JET BLACK WITH TIGHT CURLS
12-极5
DOCILE AND GENTLE
PREMIUM BUFFALO BREED FOR MILK PRODUCTION""",
        "measurements": {
            "body_length": "150-160 cm",
            "height_withers": "140-150 cm",
            "chest_width": "50-55 cm",
            "rump_angle": "5-7 degrees"
        }
    },
    "nagori": {
        "info": """ORIGINATED IN RAJASTHAN, INDIA
NA (Draft breed)
ADAPTED TO ARID CLIMATES
INDIA (Rajasthan)
MEDIUM SIZE, WHITE TO GREY COLOR
15-20
HARDY AND ACTIVE
PRIMARILY USED FOR DRAFT PURPOSES""",
        "measurements": {
            "body_length": "145-155 cm",
            "height_withers": "135-145 cm",
            "chest_width": "48-53 cm",
            "rump_angle": "6-8 degrees"
        }
    },
    "极agpuri": {
        "info": """ORIGINATED IN MAHARASHTRA, INDIA
NA (Draft breed)
ADAPTED TO TROPICAL CLIMATES
INDIA (Maharashtra)
MEDIUM SIZE, BLACK WITH WHITE MARKINGS
15-20
STRONG AND HARDY
PRIMARILY USED AS DRAFT ANIMALS""",
        "measurements": {
            "body_length": "150-160 cm",
            "height_withers": "140-150 cm",
            "chest_width": "50-55 cm",
            "rump_angle": "5-7 degrees"
        }
    },
    "nili_ravi": {
        "info": """ORIGINATED IN PUNJAB REGION OF PAKISTAN/INDIA
1500-2000 Liters
ADAPTED TO TROPICAL CLIMATES
PAKISTAN/INDIA
MEDIUM SIZE, BLACK WITH WHITE MARKINGS
12-15
DOCILE AND GENTLE
GOOD BUFFALO BREED FOR MILK PRODUCTION""",
        "measurements": {
            "body_length": "150-160 cm",
            "height_withers": "140-150 cm",
            "chest_width": "50-55 cm",
           "rump_angle": "5-7 degrees"
        }
    },
    "nimari": {
        "info": """ORIGINATED IN MADHYA PRADESH, INDIA
NA (Draft breed)
ADAPTED TO TROPICAL CLIMATES
INDIA (Madhya Pradesh)
MEDIUM SIZE, RED AND WHITE COLOR
15-20
HARDY AND ACTIVE
PRIMARILY USED FOR DRAFT PURPOSES""",
        "measurements": {
            "body_length": "145-155 cm",
            "height_withers": "135-145 cm",
            "chest_width": "48-53 cm",
            "rump_angle": "6-8 degrees"
        }
    },
    "ongole": {
        "info": """ORIGINATED IN ANDHRA PRADESH, INDIA
极A (Draft breed)
ADAPTED TO TROPICAL CLIMATES
INDIA (Andhra Pradesh)
LARGE SIZE, WHITE TO LIGHT GREY COLOR
15-20
STRONG AND HARDY
PREMIUM DRAFT BREED, EXPORTED WORLDWIDE""",
        "measurements": {
            "body_length": "155-165 cm",
            "height_withers": "145-155 cm",
            "chest_width": "52-57 cm",
            "rump_angle": "5-7 degrees"
        }
    },
    "pulikulam": {
        "info": """ORIGINATED IN TAMIL NADU, INDIA
NA (Draft breed)
ADAPTED TO TROPICAL CLIMATES
INDIA (Tamil Nadu)
SMALL SIZE, GREY TO WHITE COLOR
15-20
HARDY AND ACTIVE
PRIMARILY USED FOR DRAFT PURPOSES""",
        "measurements": {
            "body_length": "135-145 cm",
            "height_withers": "125-135 cm",
            "chest_width": "42-47 cm",
            "rump_angle": "6-8 degrees"
        }
    },
    "rathi": {
        "info": """ORIGINATED IN RAJASTHAN, INDIA
1500-2000 Liters
ADAPTED TO DESERT CLIMATES
INDIA (Rajasthan)
MEDIUM SIZE, BROWN WITH WHITE PATCHES
12-15
DOCILE AND HARDY
GOOD MILK YIELD IN ARID CONDITIONS""",
        "measurements": {
            "body_length": "150-160 cm",
            "height_withers": "极40-150 cm",
            "chest_width": "50-55 cm",
            "rump_angle": "5-极 degrees"
        }
    },
    "red_dane": {
        "info": """ORIGINATED IN DENMARK
6000-7000 Liters
ADAPTED TO TEMPERATE CLIMATES
DENMARK
LARGE SIZE, RED TO DARK RED COLOR
10-12
DOCILE AND CALM
GOOD DAIRY BREED WITH HIGH MILK YIELD""",
        "measurements": {
            "body_length": "155-165 cm",
            "height_withers": "145-155 cm",
            "chest_width": "52-57 cm",
            "rump_angle": "4-6 degrees"
        }
    },
    "red_sindhi": {
        "info": """ORIGINATED IN SINDH REGION (PAKISTAN)
1500-2000 Liters
ADAPTED TO TROPICAL CLIMATES
PAKISTAN
MEDIUM SIZE, REDDISH BROWN COLOR
12-15
DOCILE AND HARDY
GOOD MILK YIELD IN HOT CLIMATES""",
        "measurements": {
            "body_length": "150-160极",
            "height_withers": "140-150 cm",
            "chest_width": "50-55 cm",
            "rump_angle": "5-7 degrees"
        }
    },
    "sahiwal": {
        "info": """ORIGINATED IN SAHIWAL DISTRICT, PAKISTAN
2000-3000 Liters
ADAPTED TO TROPICAL CLIMATES
PAKISTAN
MEDIUM SIZE, REDDISH BROWN COLOR
12-15
DOCILE AND HARDY
ONE OF THE BEST DAIRY BREEDS IN TROPICS""",
        "measurements": {
            "body_length": "150-160 cm",
            "height_withers": "140-150 cm",
            "chest_width": "50-55 cm",
            "rump_angle": "5-7 degrees"
        }
    },
    "surti": {
        "info": """ORIGINATED IN GU极ARAT, INDIA
1200-1800 Liters
ADAPTED TO TROPICAL CLIMATES
INDIA (Gujarat)
MEDIUM SIZE, BLACK OR BROWN COLOR
12-15
DOCILE AND GENTLE
GOOD BUFFALO BREED FOR MILK PRODUCTION""",
        "measurements": {
            "body_length": "145-155 cm",
            "height_withers": "135-145 cm",
            "chest_width": "48-53 cm",
            "rump_angle": "5-7 degrees"
        }
    },
    "tharparkar": {
        "info": """ORIGINATED IN THARPARKAR DISTRICT, PAKISTAN
1500-2000 Liters
ADAPTED TO DESERT CLIMATES
PAKISTAN
MEDIUM SIZE, WHITE TO LIGHT GREY COLOR
12-15
HARDY AND DOCILE
GOOD MILK YIELD IN ARID CONDITIONS""",
        "measurements": {
            "body_length": "150-160 cm",
            "height_withers": "140-150 cm",
            "chest_width": "50-55 cm",
            "rump_angle": "5-7 degrees"
        }
    },
    "toda": {
        "info": """ORIGINATED IN NILGIRI HILLS, TAMIL NADU
NA (Draft breed)
ADAPTED TO HILLY TERRAIN
INDIA (Tamil Nadu)
MEDIUM SIZE, GREY TO BUFF COLOR
15-20
HARDY AND ACTIVE
PRIMARILY USED FOR DRAFT PURPOSES""",
        "measurements": {
            "body_length": "145-155 cm",
            "height_withers": "135-145 cm",
            "chest_width": "48-53 cm",
            "rump_angle": "6-8 degrees"
        }
    },
    "umblachery": {
        "info": """ORIGINATED IN TAMIL NADU, INDIA
NA (Draft breed)
ADAPTED TO TROPICAL CLIMATES
INDIA (Tamil Nadu)
SMALL SIZE, GREY WITH WHITE MARKINGS
15-20
HARDY AND ACTIVE
PRIMARILY USED FOR DRAFT PURPOSES""",
        "measurements": {
            "body_length": "135-145 cm",
            "height_withers": "125-135 cm",
            "chest_width": "42-47 cm",
            "rump_angle": "6-8 degrees"
        }
    },
    "vechur": {
        "info": """ORIGINATED IN KERALA, INDIA
500-800 Liters
ADAPTED TO TROPICAL CLIMATES
INDIA (Kerala)
VERY SMALL SIZE, LIGHT RED TO BROWN COLOR
极12-15
DOCILE AND GENTLE
SMALLEST CATTLE BREED, HIGH FAT MILK""",
        "measurements": {
            "body_length": "120-130 cm",
            "height_withers": "110-120 cm",
            "chest_width": "极35-40 cm",
            "rump_angle": "6-8 degrees"
        }
    }
}

IMG_SIZE = 300
CONFIDENCE_THRESHOLD = -700  # Set a reasonable confidence threshold

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to save classification data to CSV
def save_to_csv(breed, confidence, filename, timestamp):
    csv_file = "cattle_classification_data.csv"
    file_exists = os.path.isfile(csv_file)
    
    with open(csv_file, 'a', newline='') as file:
        fieldnames = ['timestamp', 'breed', 'confidence', 'filename']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            'timestamp': timestamp,
            'breed': breed,
            'confidence': confidence,
            'filename': filename
        })

# Function to display classification history
def display_classification_history():
    csv_file = "cattle_classification_data.csv"
    if os.path.isfile(csv_file):
        df = pd.read_csv(csv_file)
        st.dataframe(df)
    else:
        st.info("No classification history available yet.")

# Chatbot functionality
def chatbot_response(message):
    message = message.lower()
    
    # Greetings
    if any(word in message for word in ["hello", "hi", "hey", "hola"]):
        return "Hello! How can I help you with cattle-related questions today?"
    
    # Breed information
    elif any(word in message for word in ["breed", "identification", "identify", "type"]):
        return "You can upload an image of cattle to identify its breed using our AI model. We can identify over 40 Indian cattle breeds!"
    
    # Buying/selling
    elif any(word in message for word in ["buy", "sell", "purchase", "price", "market", "marketplace"]):
        return "You can visit our Cattle Marketplace to buy or sell cattle. Click on the 'Cattle Marketplace' button to see available listings."
    
    # Health issues
    elif any(word in message for word in ["sick", "disease", "health", "vaccine", "vaccination", "treatment"]):
        return "For health issues, I recommend consulting a veterinarian. Common cattle health concerns include foot-and-mouth disease, mastitis, and parasites. Regular vaccinations are important."
    
    # Feeding
    elif any(word in message for word in ["feed", "food", "diet", "eating", "nutrition"]):
        return "Cattle nutrition depends on age and purpose. Dairy cattle need balanced feed with proteins, energy, vitamins and minerals. Common feeds include green fodder, dry fodder, and concentrated feeds."
    
    # Milk production
    elif any(word in message for word in ["milk", "production", "yield", "lactation"]):
        return "Milk production varies by breed. High-yielding breeds like Holstein Friesian can produce 20-30 liters per day, while indigenous breeds like Gir produce 10-15 liters per day."
    
    # General care
    elif any(word in message for word in ["care", "shelter", "housing", "management"]):
        return "Proper cattle care includes clean shelter, balanced nutrition, clean water, regular health check-ups, and vaccination. Good management practices improve productivity."
    
    # Default response
    else:
        return "I'm here to help with cattle-related questions. You can ask me about breeds, buying/selling, health issues, feeding, or general care."

# Marketplace data (sample data) with additional information
marketplace_data = [
    {"name": "Gir Cow", "price": "₹65,000", "seller": "Rajesh Farms", "contact": "+91 98765 43210", "location": "Ahmedabad, Gujarat", "age": "4 years", "milk_yield": "12-15 liters/day", "lactation_stage": "2nd lactation", "vaccination": "FMD, HS, BQ vaccinated"},
    {"name": "Murrah Buffalo", "price": "₹85,000", "seller": "Singh Dairy", "contact": "+91 97654 32109", "location": "Ludhiana, Punjab", "age": "5 years", "milk_yield": "8-10 liters/day", "lactation_stage": "3rd lactation", "vaccination": "FMD, HS vaccinated"},
    {"name": "Sahiwal Cow", "price": "₹55,000", "seller": "Green Fields", "contact": "+91 96543 21098", "location": "Hisar, Haryana", "age": "3 years", "milk_yield": "10-12 liters/day", "lactation_stage": "1st lactation", "vaccination": "FMD, HS, BQ vaccinated"},
    {"name": "Jersey Cow", "price": "₹45,000", "seller": "Modern Dairy", "contact": "+91 95432 10987", "location": "Pune, Maharashtra", "age": "4 years", "milk_yield": "18-20 liters/day", "lactation_stage": "2nd lactation", "vaccination": "FMD, HS vaccinated"},
    {"name": "Tharparkar Cow", "price": "₹60,000", "seller": "Desert Cattle Co.", "contact": "+91 94321 09876", "location": "Jodh极r, Rajasthan", "age": "5 years", "milk_yield": "8-10 liters/day", "lactation_stage": "3rd lactation", "vaccination": "FMD, HS, BQ vaccinated"},
    {"name": "Holstein Friesian", "price": "₹75,000", "seller": "Elite Dairy Farms", "contact": "+91 93210 98765", "location": "Bangalore, Karnataka", "age": "3 years", "milk_yield": "22-25 liters/day", "lactation_stage": "1st lactation", "vaccination": "极MD, HS vaccinated"}
]

# Initialize session state
if 'chat_open' not in st.session_state:
    st.session_state.chat_open = False
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you with cattle-related questions today?"}]
if 'current_page' not in st.session_state:
    st.session_state.current_page = "main"
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""
if 'classification_history' not in st.session_state:
    st.session_state.classification_history = []
if 'marketplace_data' not in st.session_state:
    # Fix typo in sample data
    fixed_marketplace_data = []
    for cattle in marketplace_data:
        entry = cattle.copy()
        if "price" in entry:
            entry["price"] = entry.pop("price")
        if "location" in entry and "pune" in entry["location"]:
            entry["location"] = entry["location"].replace("pune", "")
        fixed_marketplace_data.append(entry)
    st.session_state.marketplace_data = fixed_marketplace_data

# Toggle chat function
def toggle_chat():
    st.session_state.chat_open = not st.session_state.chat_open

# Navigation function
def navigate_to(page):
    st.session_state.current_page = page

# Send message function
def send_message():
    if st.session_state.user_input.strip() != "":
        # Add user message
        st.session_state.messages.append({"role": "user", "content": st.session_state.user_input})
        
        # Get bot response
        bot_response = chatbot_response(st.session_state.user_input)
        
       # Add bot response
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        
        # Clear input
        st.session_state.user_input = ""

# Send quick reply function
def send_quick_reply(text):
    st.session_state.user_input = text
    send_message()

# Function to display breed information
def display_breed_info(breed_key, breed_data, language):
    try:
        translated_info = translate_breed_info(breed_data["info"], language)
        lines = translated_info.strip().split("\n")
        if len(lines) < 8:
            st.warning(get_translation("incomplete_info", language))
            return

        info_html = f"""
        <div class="breed-info">
            <p>🧬 <b>{get_translation("pedigree", language)}</b>: {lines[0]}</p>
            <p>🍼 <b>{get_translation("productivity", language)}</b>: {lines[1]}</p>
            <p>🌿 <b>{get_translation("rearing_conditions", language)}</b>: {lines[2]}</p>
            <p>🌍 <b>{get_translation("origin", language)}</b>: {lines[3]}</p>
            <p>🐮 <b>{get_translation("physical_chars", language)}</b>: {lines[4]}</p>
            <p>❤ <b>{get_translation("lifespan", language)}</b>: {lines[5]}</p>
            <p>💉 <b>{get_translation("temperament", language)}</b>: {lines[6]}</p>
            <p>🥩 <b>{get_translation("productivity_metrics", language)}</b>: {lines[7]}</p>
        </div>
        """
        st.markdown(info_html, unsafe_allow_html=True)

        # --- Voice Feature ---
        # Join all lines for speech
        info_text = ". ".join(lines)
        # Map Streamlit language code to gTTS language code
        lang_map = {"en": "en", "hi": "hi", "te": "te"}
        gtts_lang = lang_map.get(language, "en")
        if st.button("🔊 Speak Info"):
            tts = gTTS(text=info_text, lang=gtts_lang)
            mp3_fp = BytesIO()
            tts.write_to_fp(mp3_fp)
            st.audio(mp3_fp.getvalue(), format="audio/mp3")

        # Display physical measurements
        measurements = breed_data["measurements"]
        measurements_html = f"""
        <div class="physical-measurements">
            <h4>📏 {get_translation("physical_measurements", language)}</h4>
            <p>📏 <b>{get_translation("body_length", language)}</b>: {measurements['body_length']}</p>
            <p>📐 <b>{get_translation("height_withers", language)}</b>: {measurements['height_withers']}</p>
            <p>📊 <b>{get_translation("chest_width", language)}</b>: {measurements['chest_width']}</p>
            <p>📐 <b>{get_translation("rump_angle", language)}</b>: {measurements['rump_angle']}</p>
        </div>
        """
        st.markdown(measurements_html, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"{get_translation('info_parsing_error', language)} {str(e)}")

# Sidebar with classification history
with st.sidebar:
    st.header("📊 Classification History")
    
    csv_file = "cattle_classification_data.csv"
    if os.path.isfile(csv_file):
        df = pd.read_csv(csv_file)
        if not df.empty:
            st.dataframe(df.tail(5), use_container_width=True)
            
            # Download button
            with open(csv_file, "rb") as file:
                st.download_button(
                    label="📥 Download Full CSV",
                    data=file,
                    file_name="cattle_classification_data.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        else:
            st.info("No classification history available yet.")
    else:
        st.info("No classification history available yet.")
    
    st.markdown("---")
    st.header("ℹ About")
    st.info("This app identifies Indian cattle breeds using AI and provides information about each breed's characteristics and physical measurements.")

# Main app content
if st.session_state.current_page == "main":
    # Streamlit UI
    st.markdown(f'<h1 class="main-header">{get_translation("title", language)}</h1>', unsafe_allow_html=True)
    st.markdown(f'<h2 class="sub-header">{get_translation("subtitle", language)}</h2>', unsafe_allow_html=True)

    st.info(get_translation("upload_info", language))

    # Add marketplace button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(get_translation("marketplace_button", language), use_container_width=True):
            navigate_to("marketplace")

    # Image uploader
    st.markdown(f"{get_translation('upload_label', language)}")

    # Create a custom file uploader area
    uploaded_file = None
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            # Custom upload area
            st.markdown(
                f"""
                <div style="border: 2px dashed #ccc; border-radius: 5px; padding: 20px; text-align: center; margin: 10px 0;">
                    <p style="font-weight: bold; margin-bottom: 10px;">{get_translation('drag_drop', language)}</p>
                    <p style="font-size: 12px; color: #666; margin-bottom: 15px;">{get_translation('file_limit', language)}</p>
                    <hr style="border-top: 1px solid #eee; margin: 15px 0;">
                    <p style="font-size: 14px; margin-top: 15px;">{get_translation('browse_files', language)}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Hidden file uploader
            uploaded_file = st.file_uploader(
                "",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed",
                help=get_translation("help_text", language)
            )

    # Prediction function for PyTorch
    def predict_breed(image):
        try:
            # Apply transformations
            image = transform(image).unsqueeze(0).to(device)
            
            # Make prediction
            with torch.no_grad():
                outputs = model(image)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted_idx = torch.max(probabilities, 0)
                predicted_label = breed_labels[predicted_idx.item()]
                confidence_percent = confidence.item() * 100
                
            return predicted_label, confidence_percent
        except Exception as e:
            st.error(f"{get_translation('prediction_error', language)}: {str(e)}")
            return None, 0

    # Handle image and prediction
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption=get_translation("image_caption", language), use_container_width=True)

            with st.spinner(get_translation("analyzing", language)):
                breed, confidence = predict_breed(image)

            if breed is None:
                st.error(get_translation("prediction_error", language))
            elif confidence < CONFIDENCE_THRESHOLD:
                st.error(get_translation("confidence_error", language))
            else:
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.markdown(f'<p class="prediction-text">{get_translation("predicted_breed", language)} <b>{breed}</b></p>', unsafe_allow_html=True)
                #st.markdown(f'<p class="confidence-text">{get_translation("confidence", language)}: {confidence:.2f}%</p>', unsafe_allow_html=True)#
                st.markdown('</div>', unsafe_allow_html=True)

                # Save to CSV
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                filename = uploaded_file.name
                save_to_csv(breed, f"{confidence:.2f}%", filename, timestamp)
                
                # Add to session state
                st.session_state.classification_history.append({
                    "timestamp": timestamp,
                    "breed": breed,
                    "confidence": f"{confidence:.2f}%",
                    "filename": filename
                })

                breed_key = breed.lower().strip()
                if breed_key in breed_info_raw:
                    st.subheader(get_translation("breed_info", language))
                    display_breed_info(breed_key, breed_info_raw[breed_key], language)
                else:
                    st.warning(get_translation("no_info", language))

        except Exception as e:
            st.error(f"{get_translation('processing_error', language)} {str(e)}")

    # Add footer
    st.markdown("---")
    st.markdown(
        f"""
        <div class="footer">
            <p>{get_translation("refresh", language)}</p>
            <p>{get_translation("heritage", language)}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Marketplace page
elif st.session_state.current_page == "marketplace":
    st.markdown(f'<h1 class="Main-header">{get_translation("marketplace_title", language)}</h1>', unsafe_allow_html=True)
    st.markdown(f'<h2 class="sub-header">{get_translation("marketplace_subtitle", language)}</h2>', unsafe_allow_html=True)
    
    # Back button
    if st.button(get_translation("back_button", language)):
        navigate_to("main")
    
    st.info(get_translation("marketplace_info", language))
    
    # Display marketplace listings with additional information
    for cattle in st.session_state.marketplace_data:
        st.markdown(
            f"""
            <div class="cattle-card">
                <div class="cattle-name">{cattle.get('name','')}</div>
                <div class="cattle-price">{get_translation("price", language)}: {cattle.get('price','')}</div>
                <div class="seller-info">{get_translation("age", language)}: {cattle.get('age','')}</div>
                <div class="seller-info">{get_translation("milk_yield", language)}: {cattle.get('milk_yield','')}</div>
                <div class="seller-info">{get_translation("lactation_stage", language)}: {cattle.get('lactation_stage','')}</div>
                <div class="seller-info">{get_translation("vaccination", language)}: {cattle.get('vaccination','')}</div>
                <div class="seller-info">{get_translation("seller", language)}: {cattle.get('seller','')}</div>
                <div class="seller-info">{get_translation("contact", language)}: {cattle.get('contact','')}</div>
                <div class="seller-info">{get_translation("location", language)}: {cattle.get('location','')}</div>
                {"<div class='seller-info'><b>" + get_translation("description", language) + ":</b> " + cattle.get('description','') + "</div>" if cattle.get('description') else ""}
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Add your listing section
    st.markdown("---")
    st.subheader(get_translation("add_listing", language))
    
    with st.form("add_listing"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input(get_translation("cattle_breed", language))
            price = st.text_input(get_translation("price", language))
            age = st.text_input(get_translation("age", language))
        with col2:
            seller = st.text_input(get_translation("seller", language))
            contact = st.text_input(get_translation("contact", language))
            milk_yield = st.text_input(get_translation("milk_yield", language))
        
        location = st.text_input(get_translation("location", language))
        lactation_options = ["1st lactation", "2nd lactation", "3rd lactation", "4th lactation+"]
        lactation_stage = st.selectbox(get_translation("lactation_stage", language), lactation_options)
        vaccination_options = ["FMD vaccinated", "HS vaccinated", "BQ vaccinated", "None"]
        vaccination = st.multiselect(get_translation("vaccination", language), vaccination_options)
        description = st.text_area(get_translation("description", language))
        
        submitted = st.form_submit_button(get_translation("submit_listing", language))
        if submitted:
            st.session_state.marketplace_data.append({
                "name": name,
                "price": price,
                "age": age,
                "seller": seller,
                "contact": contact,
                "milk_yield": milk_yield,
                "location": location,
                "lactation_stage": lactation_stage,
                "vaccination": ", ".join(vaccination),
                "description": description
            })
            st.success(get_translation("listing_submitted", language))
            st.experimental_rerun()

# Chat toggle button - placed in the bottom right corner
st.markdown(
    """
    <div style="position: fixed; bottom: 20px; right: 20px; z-index: 1000;">
    """, 
    unsafe_allow_html=True
)

if st.button("💬", key="chat_toggle", help="Chat with us"):
    toggle_chat()

st.markdown("</div>", unsafe_allow_html=True)

# Chat interface - appears in the bottom right when toggled
if st.session_state.chat_open:
    st.markdown(
        f"""
        <div style='position: fixed; bottom: 90px; right: 20px; width: 350px; height: 450px; 
                    background-color: white; border-radius: 15px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15); 
                    z-index: 1000; display: flex; flex-direction: column; overflow: hidden;'>
            <div style='background-color: #3498db; color: white; padding: 15px; font-weight: bold; 
                        border-top-left-radius: 15px; border-top-right-radius: 15px;'>
                {get_translation("chat_title", language)} 💬
            </div>
            <div style='flex: 1; padding: 15px; overflow-y: auto; display: flex; flex-direction: column; gap: 10px;'>
        """, 
        unsafe_allow_html=True
       )
    
    # Display messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(
                f"""
                <div style='max-width: 80%; padding: 10px 15px; border-radius: 15px; margin-bottom: 10px;
                            background-color: #3498db; color: white; align-self: flex-end; border-bottom-right-radius: 5px;'>
                    {message["content"]}
                </div>
                """, 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style='max-width: 80%; padding: 10px 15px; border-radius: 15px; margin-bottom: 10px;
                            background-color: #f1f1f1; align-self: flex-start; border-bottom-left-radius: 5px;'>
                    {message["content"]}
                </div>
                """, 
                unsafe_allow_html=True
            )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Quick replies
    st.markdown(
        """
        <div style='display: flex; flex-wrap: wrap; gap: 5px; padding: 10px; background-color: #f9f9f9;'>
            <div style='background-color: #e8f4f8; border: 1px solid #3498db; border-radius: 15px; 
                        padding: 5px 10px; font-size: 12px; cursor: pointer;' 
                 onclick='window.parent.document.querySelector("input[placeholder=\\"Type your message...\\"]").value = "How to identify cattle breed?"'>
                Identify breed
            </div>
            <div style='background-color: #e8f4f8; border: 1px solid #3498db; border-radius: 15px; 
                        padding: 5px 10px; font-size: 12px; cursor: pointer;' 
                 onclick='window.parent.document.querySelector("input[placeholder=\\"Type your message...\\"]").value = "How to buy cattle?"'>
                Buying cattle
            </div>
            <极div style='background-color: #e8f4f8; border: 1px solid #3498db; border-radius: 15px; 
                        padding: 5px 10px; font-size: 12px; cursor: pointer;' 
                 onclick='window.parent.document.querySelector("input[placeholder=\\"Type your message...\\"]").value = "Common health issues?"'>
                Health issues
            </div>
            <div style='background-color: #e8f4f8; border: 1px solid #3498db; border-radius: 15px; 
                        padding: 5px 10px; font-size: 12px; cursor: pointer;' 
                 onclick='window.parent.document.querySelector("极nput[placeholder=\\"Type your message...\\"]").value = "Feeding recommendations?"'>
                Feeding
            </div>
        </div>
        <div style='display: flex; padding: 10px; border-top: 1px solid #ddd; background-color: white;'>
            <input type='text' placeholder='Type your message...' 
                   style='flex: 1; padding: 10px; border: 1px solid #ddd; border-radius: 20px; outline: none;'
                   value='""" + st.session_state.user_input + """'
                   onkeypress='if(event.key==="Enter") {window.parent.document.querySelector("button[title=\\"Send message\\"]").click()}'>
            <button style='margin-left: 10px; background-color: #3498db; color: white; border: none; 
                           border-radius: 20px; padding: 10px 15px; cursor: pointer;'
                    onclick='window.parent.document.querySelector("button[title=\\"Send message\\"]").click()'>
                Send
            </button>
        </div>
        </div>
        """, 
        unsafe_allow_html=True
    )

# Chat input form (outside the chat container)
# """with st.form("chat_input", clear_on_submit=True):
#     user_input = st.text_input("Type your message...", key="user_input", label_visibility="collapsed")
#     submitted = st.form_submit_button("Send message", use_container_width=True)
#     if submitted and user_input.strip() != "":
#         send_message()"""