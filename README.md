🐄 Indian Cattle Breed Identifier

An AI-powered web application for identifying Indian cattle breeds from images, providing detailed breed information, and facilitating cattle trading through a built-in marketplace.



Features

🎯 Core Functionality

>Breed Identification: Upload images of cattle to identify their breed using a deep learning model

>Multi-language Support: Available in English, Hindi, and Telugu

>Breed Information: Detailed characteristics, productivity metrics, and physical measurements for each breed

> Classification History: Track and download past identification results



🛒 Marketplace



Buy/Sell Cattle: Browse and create listings for cattle trading

Detailed Listings: Includes price, age, milk yield, vaccination status, and seller contact information

Location-based: Filter by geographical location



💬 Interactive Assistant

Chatbot Support: Get answers to cattle-related questions

Quick Replies: Common questions with one-click responses

Multi-lingual: Chat support in all available languages




🔊 Audio Features
Text-to-Speech: Listen to breed information descriptions

Multi-language Audio: Supports English, Hindi, and Telugu audio output




Supported Breeds
The application can identify 41 different Indian cattle breeds:

Alambadi, Amritmahal, Ayrshire, Banni, Bargur, Bhadawari, Brown Swiss, Dangi, Deoni, Gir, Guernsey, Hallikar, Hariana, Holstein Friesian, Jaffrabadi, Jersey, Kangayam, Kankrej, Kasargod, Kenkatha, Kherigarh, Khillari, Krishna Valley, Malnad Gidda, Mehsana, Murrah, Nagori, Nagpuri, Nili Ravi, Nimari, Ongole, Pulikulam, Rathi, Red Dane, Red Sindhi, Sahiwal, Surti, Tharparkar, Toda, Umblachery, Vechur




Technical Details



Model Architecture
Backbone: ResNet-50

Input Size: 300×300 pixels

Framework: PyTorch

Training: Custom-trained on Indian bovine breeds dataset




Dependencies
text
streamlit
torch
torchvision
numpy
Pillow
timm
gTTS
pandas





File Structure
text

cattle-breed-identifier/
├── app.py                 # Main application file
├── best_resnet50_indian_bovine_breeds.pth  # Trained model weights
├── cattle_classification_data.csv  # Classification history
└── requirements.txt       # Python dependencies



Installation & Setup
Clone the repository

bash
git clone https://github.com/anris18/Cattle.git
cd Cattle
Install dependencies

bash
pip install -r requirements.txt
Download model weights

Ensure best_resnet50_indian_bovine_breeds.pth is in the project directory

Or train your own model using the provided architecture

Run the application

bash
streamlit run app.py
Access the application

Open your browser and navigate to http://localhost:8501




Usage
Breed Identification
Click "Choose a cattle image" or drag and drop an image

Wait for the AI model to analyze the image

View the predicted breed with confidence score

Explore detailed breed information and physical measurements




Marketplace
Click the "Cattle Marketplace" button

Browse existing listings or create your own

Contact sellers directly using provided contact information

Add new listings with detailed specifications




Data Collection
The application automatically saves classification history to (https://www.kaggle.com/code/nishanthmatthewpaul/cow-sih) with:


Customization
Adding New Breeds
Add the breed name to the breed_labels list

Create breed information in the breed_info_raw dictionary

Include physical measurements for the new breed



Adding New Languages
Extend the translations dictionary with new language support

Add language-specific breed information translations

Update the language selector UI




Performance Notes
The model works best with clear, side-profile images of cattle

Confidence threshold is set to ensure accurate predictions

Image uploads are limited to 200MB per file (JPG, JPEG, PNG formats)




Contributing
We welcome contributions to:

Improve model accuracy

Add support for more breeds

Translate to additional languages

Enhance marketplace features

Optimize UI/UX design



License
This project is licensed under the MIT License - see the LICENSE file for details.




Streamlit for the excellent web application framework

Support
For questions or support, please use the in-app chat feature or create an issue in the GitHub repository.

Celebrating India's rich bovine heritage 🐄
