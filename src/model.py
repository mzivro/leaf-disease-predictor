from sklearn.preprocessing import LabelEncoder
from torchvision.transforms import transforms
from PIL import Image

import torch.nn as nn
import torch


class NoImageError(Exception):
    """Exception raised when no image is provided for prediction."""

    def __init__(self) -> None:
        super().__init__("No image found in request!")


class PredictorConv(nn.Module):
    """Convolutional neural network used for leaf disease classification.

    Parameters
    ----------
    labels_count : int
        Number of output classes supported by the model.
    """

    def __init__(self, labels_count: int) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pooling = nn.MaxPool2d(2, 2)

        self.relu = nn.ReLU()

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128 * 16 * 16, 128)

        self.output = nn.Linear(128, labels_count)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the convolutional neural network.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (batch_size, 3, 128, 128).

        Returns
        -------
        torch.Tensor
            Output logits for each classification label, with shape
            (batch_size, labels_count).
        """
        x = self.conv1(x)  # (32, 128, 128)
        x = self.pooling(x)  # (32, 64, 64)
        x = self.relu(x)

        x = self.conv2(x)  # (64, 64, 64)
        x = self.pooling(x)  # (64, 32, 32)
        x = self.relu(x)

        x = self.conv3(x)  # (128, 32, 32)
        x = self.pooling(x)  # (128, 16, 16)
        x = self.relu(x)

        x = self.flatten(x)
        x = self.linear(x)
        x = self.output(x)

        return x


class Model:
    """Load and run inference using a trained leaf disease classification model.

    Parameters
    ----------
    checkpoint_path : str
        Path to the model checkpoint containing the model weights and
        classification metadata.
    """

    def __init__(self, checkpoint_path: str) -> None:
        # set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Using device: {self.device}")

        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        # set transform pipeline
        self.transform = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float),
            ]
        )

        # load checkpoint
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )

        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = checkpoint["classes"]

        # load model and weights
        self.net = PredictorConv(checkpoint["classes_count"])

        self.net.load_state_dict(
            checkpoint["model_state_dict"],
        )

        self.net.to(self.device)

        self.net.eval()

        print("Model ready")

    def __del__(self) -> None:
        """Release CUDA resources used by the model."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def predict(self, image: Image.Image) -> str:
        """Predict the disease class for a leaf image.

        Parameters
        ----------
        image : PIL.Image.Image
            Input image containing a plant leaf.

        Returns
        -------
        str
            Predicted disease class label.

        Raises
        ------
        NoImageError
            If image is not provided.
        """
        if not image:
            raise NoImageError()

        image = self.transform(image).to(self.device)

        with torch.no_grad():
            output = self.net(image.unsqueeze(0))

        output = torch.argmax(output, axis=1).item()

        return self.label_encoder.inverse_transform([output])[0]
