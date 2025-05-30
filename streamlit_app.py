import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


st.set_page_config(
    page_title="Cat vs Dog Classifier 🐱🐶",
    page_icon="🐾",
    layout="centered",  # You can also use "wide"
    initial_sidebar_state="auto"
)

# Load model
model = tf.keras.models.load_model("model.h5")
class_names = ['Cat', 'Dog']

st.title("🐱🐶 Cat vs Dog Classifier")
st.write("Upload an image of a cat or dog to see the prediction.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Resize the image to (160, 160)
    img = image.resize((160, 160))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize

    # Predict
    prediction = model.predict(img_array)[0][0]

    label = "🐱 Cat" if prediction < 0.5 else "🐶 Dog"
    confidence = 100 * (1 - prediction if prediction < 0.5 else prediction)

    st.markdown(f"### Prediction: {label}")
    st.markdown(f"**Confidence:** {confidence:.2f}%")

    if confidence < 70:
        st.warning("⚠️ Low confidence — model may be guessing. Try with clearer images or train with more data.")
