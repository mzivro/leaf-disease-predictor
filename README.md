# Leaf Disease Predictor

A deep learning application for **plant leaf disease classification**. The project provides a REST API built with **FastAPI** and a web interface built with **Streamlit**.

The application supports multiple plant species and uses a dedicated convolutional neural network model for each species. Trained models are automatically downloaded from Hugging Face when the API starts.

Try it out: https://mzivro-leaf-disease-pred.streamlit.app/

## Features

* Leaf disease classification for 9 plant species
* Custom convolutional neural network built with PyTorch
* FastAPI REST API for model inference
* Streamlit web interface
* Automatic model downloading from Hugging Face Hub
* Separate model for each supported plant species
* Automatic CUDA support when a compatible GPU is available
* Pydantic request and response validation
* Health and readiness endpoints
* Notebook with training process included
* Demo app included

## Technologies

* PyTorch
* Torchvision
* FastAPI
* Pydantic
* Streamlit
* Pillow
* scikit-learn
* Huggign Face Hub
* Requests

## Supported Species and model accuracy

| Species       | Detected diseases                                                                         | Accuracy [%] |
| ------------- | ------------------------------------------------------------------------------------------|--------------|
| Apple         | Apple Scab - Black Rot - Cedar Apple Rust                                                 | 89.8         |
| Bell Pepper   | Bacterial Spot                                                                            | 98           |
| Cherry        | Powdery Mildew                                                                            | 100          |
| Corn (Maize)  | Cercospora Leaf Spot - Common Rust - Northern Leaf Blight                                 | 91.8         |
| Grape         | Black Rot - Esca (Black Measles) - Leaf Blight                                            | 93.4         |
| Peach         | Bacterial Spot                                                                            | 100          |
| Potato        | Early Blight - Late Blight                                                                | 97.2         |
| Strawberry    | Leaf Scorch                                                                               | 98.9         |
| Tomato        | Bacterial Spot - Early Blight - Late Blight - Septoria Leaf Spot - Yellow Leaf Curl Virus | 87.5         |

## Project Architecture

The application consists of three main components:

```text
                         ┌──────────────────────┐
                         │      Streamlit       │
                         │       Client         │
                         └──────────┬───────────┘
                                    │
                              HTTP POST /predict
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │       FastAPI        │
                         │        Server        │
                         └──────────┬───────────┘
                                    │
                              Select model
                              by species
                                    │
                                    ▼
                    ┌──────────────────────────────┐
                    │       PyTorch Model          │
                    │                              │
                    │  Image → CNN → Prediction    │
                    └──────────────────────────────┘
                                    ▲
                                    │
                            Hugging Face Hub
```

### Prediction flow

1. The user uploads a leaf image through the Streamlit client.
2. The client encodes the image as Base64.
3. The selected plant species and encoded image are sent to the FastAPI server.
4. The server decodes the image into a PIL image.
5. The model associated with the selected species is used for inference.
6. The CNN processes the image and produces class logits.
7. The predicted class is decoded using the stored `LabelEncoder`.
8. The prediction is returned to the Streamlit client.

## Machine Learning Model

The classifier is implemented using PyTorch.

The convolutional neural network consists of:

```text
Input: 3 × 128 × 128
        │
        ▼
Conv2D: 3 → 32
        │
      MaxPool
        │
      ReLU
        │
        ▼
Conv2D: 32 → 64
        │
      MaxPool
        │
      ReLU
        │
        ▼
Conv2D: 64 → 128
        │
      MaxPool
        │
      ReLU
        │
        ▼
Flatten
        │
        ▼
Linear: 32768 → 128
        │
        ▼
Linear: 128 → number of classes
        │
        ▼
Prediction
```

Before inference, images are transformed using the following pipeline:

```text
Resize → 128 × 128
   ↓
ToTensor
   ↓
ConvertImageDtype(torch.float)
```

The model automatically uses CUDA when available:

```python
torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Otherwise, inference is performed on the CPU.

## Installation

Clone the repository:

```bash
git clone https://github.com/mzivro/leaf-disease-predictor.git
cd leaf-disease-predictor
```

Create and activate a virtual environment:

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

The application consists of two processes:

* FastAPI backend
* Streamlit frontend

### 1. Start the API

From the project root:

```bash
fastapi run src/server.py
```

The API will be available at:

```text
http://localhost:8000
```

On startup, the application downloads the required model checkpoints from Hugging Face Hub and loads them into memory.

### 2. Start the Streamlit client

In a separate terminal:

```bash
streamlit run src/client.py
```

The Streamlit application should open automatically in your browser.

If it does not, open:

```text
http://localhost:8501
```

## API

### `GET /health`

Checks whether the API process is running.

Example response:

```json
{
  "status": "ok"
}
```

### `GET /ready`

Checks whether the machine learning models have been loaded.

Example response:

```json
{
  "status": "ready"
}
```

If the models are not loaded, the endpoint returns:

```text
503 Service Unavailable
```

### `POST /predict`

Performs leaf disease classification.

#### Request

```json
{
  "species": "Apple",
  "image_encoded": "<base64-encoded-image>"
}
```

#### Response

```json
{
  "prediction": "Apple___Apple_scab"
}
```

The exact prediction labels depend on the classes stored in the corresponding model checkpoint.

## API Validation

The API validates the prediction request before performing inference.

### Empty image

If `image_encoded` is empty:

```json
{
  "detail": "No image!"
}
```

The API returns HTTP `400`.

### Unsupported species

If the requested species is not supported:

```json
{
  "detail": "Species unsupported"
}
```

The API returns HTTP `400`.

## Dataset

Dataset used for training these models is **Plant Village Dataset (Updated)** from **Kaggle**.

Test images comes also from this dataset.

This dataset is licensed under the **CC0: Public Domain** license.

Dataset source:
https://www.kaggle.com/datasets/tushar5harma/plant-village-dataset-updated/

License:
https://creativecommons.org/publicdomain/zero/1.0/

## License

MIT License. Feel free to use, modify, and build upon this project.
