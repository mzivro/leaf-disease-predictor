from src.schemas import PredictionRequest, PredictionResponse
from huggingface_hub import hf_hub_download
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from src.model import Model
from PIL import Image

import base64
import io


@asynccontextmanager
async def lifespan(app: FastAPI) -> None:
    """Initialize and release machine learning models for the application.

    Models for all supported plant species are downloaded from Hugging Face
    and loaded during application startup. The models are released when the
    application shuts down.

    Parameters
    ----------
    app : FastAPI
        FastAPI application instance whose state is initialized with the
        loaded models.

    Yields
    ------
    None
        Control is returned to the FastAPI application after model
        initialization.
    """
    app.state.models = {
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

    app.state.classes = app.state.models.keys()

    yield

    del app.state.models


app = FastAPI(title="Leaf Disease Predictor API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    """Check whether the API server is running.

    Returns
    -------
    dict
        Dictionary containing the server status.
    """
    return {
        "status": "ok",
    }


@app.get("/ready")
def ready() -> dict:
    """Check whether the machine learning models are loaded.

    Returns
    -------
    dict
        Dictionary containing the readiness status.

    Raises
    ------
    HTTPException
        If the models have not been loaded. Returns HTTP status code 503.
    """
    if not hasattr(app.state, "models"):
        raise HTTPException(
            status_code=503,
            detail="Models not loaded",
        )

    return {"status": "ready"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict a leaf disease from a base64-encoded image.

    Parameters
    ----------
    request : PredictionRequest
        Request containing the plant species and base64-encoded image.

    Returns
    -------
    PredictionResponse
        Response containing the predicted disease class.

    Raises
    ------
    HTTPException
        If the image is empty, with HTTP status code 400.
    HTTPException
        If the requested plant species is not supported, with HTTP status
        code 400.
    """
    if not request.image_encoded.strip():
        raise HTTPException(status_code=400, detail="No image!")

    if request.species not in app.state.classes:
        raise HTTPException(status_code=400, detail="Species unsupported")

    # decode image into PIL image
    image_bytes = base64.b64decode(request.image_encoded)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # send to chosen model
    model = app.state.models[request.species]
    prediction = model.predict(image)

    return PredictionResponse(prediction=prediction)
