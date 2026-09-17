from huggingface_hub import hf_hub_download
from src.model import Model
from PIL import Image

import streamlit as st
import requests

if "models" not in st.session_state:
    st.session_state.models = {
        "Apple": Model(
            hf_hub_download(
                repo_id="mzivro/my-models",
                filename="apple_leaf_disease_predictor.pth",
                repo_type="model",
            )
        ),
        "Bell Pepper": Model(
            hf_hub_download(
                repo_id="mzivro/my-models",
                filename="bell_pepper_leaf_disease_predictor.pth",
                repo_type="model",
            )
        ),
        "Cherry": Model(
            hf_hub_download(
                repo_id="mzivro/my-models",
                filename="cherry_leaf_disease_predictor.pth",
                repo_type="model",
            )
        ),
        "Corn (Maize)": Model(
            hf_hub_download(
                repo_id="mzivro/my-models",
                filename="corn_leaf_disease_predictor.pth",
                repo_type="model",
            )
        ),
        "Grape": Model(
            hf_hub_download(
                repo_id="mzivro/my-models",
                filename="grape_leaf_disease_predictor.pth",
                repo_type="model",
            )
        ),
        "Peach": Model(
            hf_hub_download(
                repo_id="mzivro/my-models",
                filename="peach_leaf_disease_predictor.pth",
                repo_type="model",
            )
        ),
        "Potato": Model(
            hf_hub_download(
                repo_id="mzivro/my-models",
                filename="potato_leaf_disease_predictor.pth",
                repo_type="model",
            )
        ),
        "Strawberry": Model(
            hf_hub_download(
                repo_id="mzivro/my-models",
                filename="strawberry_leaf_disease_predictor.pth",
                repo_type="model",
            )
        ),
        "Tomato": Model(
            hf_hub_download(
                repo_id="mzivro/my-models",
                filename="tomato_leaf_disease_predictor.pth",
                repo_type="model",
            )
        ),
    }

if "species_list" not in st.session_state:
    st.session_state.species_list = st.session_state.models.keys()

st.title("Leaf Disease Predictor - Demo")

uploaded_file = st.file_uploader(
    "Choose image",
    type=["jpg", "jpeg", "png", "webp", "bmp", "tiff"],
    max_upload_size=10,
)

if uploaded_file is not None:
    st.image(Image.open(uploaded_file), width="stretch")

species = st.selectbox("Choose species", st.session_state.species_list)

if st.button("Predict", width="stretch"):
    if uploaded_file is None:
        st.error("No uploaded image")
    else:
        prediction = st.session_state.models[species].predict(Image.open(uploaded_file))

        st.success(f"Prediction: {prediction}")
