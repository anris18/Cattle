import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
import os
from tensorflow.keras.callbacks import Callback

# ====== SETTINGS ======
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 6
TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"
MODEL_PATH = "cattle_breed_model_clean.h5"
CONFIDENCE_THRESHOLD = 60.0

st.set_page_config(page_title="🐄 Cattle Breed Identifier", layout="centered")
st.title("🐄 Cattle Breed Identifier")
st.write("Upload an image to predict its breed, retrain the model, or download the trained model.")

# ====== Load or create model ======
@st.cache_resource(show_spinner=False)
def load_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    else:
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
        base_model.trainable = False
        inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
        model = models.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

model = load_model()

# ====== Breed info ======
breed_info_raw = {
    "ayrshire": "DEVELOPED IN AYRSHIRE\n4500 Liters\nTemperate\nScotland\nMedium, Red-White\n8\nAlert\nHigh milk quality",
    "friesian": "FROM NETHERLANDS\n6500 Liters\nTemperate\nNetherlands\nLarge, Black-White\n13\nDocile\nMilk + Draught",
    "jersey": "BRITISH BREED\n5500 Liters\nWarm\nScotland\nSmall-Medium, Light brown\n10\nFriendly\nHigh butterfat",
    "lankan white": "ZEBU+EUROPEAN\n4331 Liters\nTemperate\nSri Lanka\nMedium, Zebu traits\n12\nCalm\nHigh milk yield",
    "sahiwal": "SAHIWAL, PAKISTAN\n3000 Liters\nTropical\nPakistan\nMedium, Reddish brown\n6\nCalm\nModerate milk yield",
    "zebu": "ZEBU+EUROPEAN\n4000 Liters\nTropical\nAustralia\nMedium, Zebu traits\n10\nDocile\nModerate milk yield"
}
breed_info = {k.lower(): v for k, v in breed_info_raw.items()}
breed_labels = ["Ayrshire", "Friesian", "Jersey", "Lankan White", "Sahiwal", "Zebu"]

# ====== Image Upload & Predict ======
uploaded_file = st.file_uploader("Choose a cattle image", type=["jpg", "jpeg", "png"])

def predict_breed(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0]
    predicted_label = breed_labels[np.argmax(prediction)]
    confidence = float(np.max(prediction)) * 100
    return predicted_label, confidence

def display_breed_info(breed_key):
    info = breed_info.get(breed_key.lower())
    if info:
        st.text(info)
    else:
        st.warning("No info found.")

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='📷 Uploaded Image', use_container_width=True)
        with st.spinner("Predicting breed..."):
            breed, confidence = predict_breed(image)
        if confidence < CONFIDENCE_THRESHOLD:
            st.error("🚫 Low confidence. Try another image.")
        else:
            st.success(f"✅ Breed: {breed} ({confidence:.2f}%)")
            display_breed_info(breed)
    except Exception as e:
        st.error(f"Error: {e}")

# ====== Retrain Section ======
st.subheader("🔄 Retrain Model")
epochs = st.number_input("Select number of epochs", min_value=1, max_value=50, value=5, step=1)

if st.button("Start Retraining"):
    progress_bar = st.progress(0)
    status_text = st.empty()

    class StreamlitProgressCallback(Callback):
        def on_epoch_end(self, epoch, logs=None):
            progress = (epoch + 1) / epochs
            progress_bar.progress(progress)
            status_text.text(f"Epoch {epoch+1}/{epochs} - Accuracy: {logs.get('accuracy'):.4f} - Val Accuracy: {logs.get('val_accuracy'):.4f}")

    with st.spinner("Training model..."):
        train_gen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True
        ).flow_from_directory(TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE),
                             batch_size=BATCH_SIZE, class_mode='categorical')

        val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
            VAL_DIR, target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE, class_mode='categorical'
        )

        base_model = tf.keras.applications.MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                                       include_top=False, weights='imagenet')
        base_model.trainable = False
        inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
        new_model = models.Model(inputs, outputs)
        new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        new_model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=[StreamlitProgressCallback()])
        new_model.save(MODEL_PATH)
        st.success("✅ Model retrained and saved!")
        st.balloons()
        st.cache_resource.clear()
        model = load_model()
        st.info("🔄 Model reloaded. Ready for new predictions!")

# ====== Download Model Button ======
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        st.download_button(
            label="💾 Download Trained Model",
            data=f,
            file_name="cattle_breed_model_clean.h5",
            mime="application/octet-stream"
        )
