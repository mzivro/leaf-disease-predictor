from pydantic import BaseModel


class PredictionRequest(BaseModel):
    """Request schema for the prediction endpoint.

    Parameters
    ----------
    species : str
        Name of the plant species for which the prediction is requested.
    image_encoded : str
        Base64-encoded image data.
    """

    species: str
    image_encoded: str


class PredictionResponse(BaseModel):
    """Response schema returned by the prediction endpoint.

    Parameters
    ----------
    prediction : str
        Predicted disease class label.
    """

    prediction: str
