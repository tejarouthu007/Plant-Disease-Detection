import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
import gdown

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.keras"

drive_file_id = "1--JDi46vVyMu3KLwnFCcdh78bX8e-YoI"

if not os.path.exists(model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    gdown.download(f"https://drive.google.com/uc?id={drive_file_id}", model_path, quiet=False)

# load the model
model = tf.keras.models.load_model(model_path)

# Load class names
class_indices_path = f"{working_dir}/class_indices.json"
if os.path.exists(class_indices_path):
    with open(class_indices_path, "r") as f:
        class_indices = json.load(f)
else:
    st.error("Error: class_indices.json not found!")
    st.stop()

# Function to preprocess the image
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array.astype('float32') / 255.  # Normalize
    return img_array

# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices.get(str(predicted_class_index), "Unknown")
    return predicted_class_name

# Streamlit App
st.title('🌱 Plant Disease Detector')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            # Preprocess and predict
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f'Prediction: {prediction}')
