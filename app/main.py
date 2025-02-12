import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
import gdown
from tensorflow.keras import backend as K

# Force TensorFlow to use CPU (Streamlit Cloud has no GPU support)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Define paths
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/model/plant_disease_prediction_model.keras"
class_indices_path = f"{working_dir}/class_indices.json"
drive_file_id = "1mtuzs9rdLIr-H5Bt_Ow5jbppFK9gPmCF"

# Download the model if it doesn't exist
if not os.path.exists(model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    gdown.download(f"https://drive.google.com/uc?id={drive_file_id}", model_path, quiet=False)

# Cache the model loading to avoid reloading on every prediction
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(model_path)

model = load_model()

# Load class names
if os.path.exists(class_indices_path):
    with open(class_indices_path, "r") as f:
        class_indices = json.load(f)
else:
    st.error("Error: class_indices.json not found!")
    st.stop()

# Function to preprocess the image
def load_and_preprocess_image(image, target_size=(224, 224)):
    img = image.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array.astype('float32') / 255.  # Normalize
    return img_array

# Function to Predict the Class of an Image
def predict_image_class(model, image, class_indices):
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices.get(str(predicted_class_index), "Unknown")
    K.clear_session()  # Clear session to free memory
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
            prediction = predict_image_class(model, image, class_indices)
            st.success(f'Prediction: {prediction}')
