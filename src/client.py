from PIL import Image

import streamlit as st
import requests
import base64

API_URL = "http://localhost:8000/predict"

species_list = [
    "Apple",
    "Bell Pepper",
    "Cherry",
    "Corn (Maize)",
    "Grape",
    "Peach",
    "Potato",
    "Strawberry",
    "Tomato",
]

st.title("Leaf Disease Predictor - Client")

uploaded_file = st.file_uploader(
    "Choose image",
    type=["jpg", "jpeg", "png", "webp", "bmp", "tiff"],
    max_upload_size=10,
)

if uploaded_file is not None:
    st.image(Image.open(uploaded_file), width="stretch")

species = st.selectbox("Choose species", species_list)

if st.button("Predict", width="stretch"):
    if uploaded_file is None:
        st.error("No uploaded image")
    else:
        # encode image into base64
        image_bytes = uploaded_file.getvalue()
        image_encoded = base64.b64encode(image_bytes).decode("utf-8")

        payload = {
            "image_encoded": image_encoded,
            "species": species,
        }

        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            prediction = response.json()["prediction"]

            st.success(f"Prediction: {prediction}")
        else:
            st.error(response.text)
