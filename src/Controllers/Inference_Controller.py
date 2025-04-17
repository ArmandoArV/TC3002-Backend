import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F  # Import for softmax
import os


class InferenceController:
    def __init__(self, model_filename="vgg11_15_epochs_aug_RMS.pt", device=None):
        """
        Initialize the inference controller.
        - Loads the model from the correct path.
        - Uses the available device (GPU or CPU).
        """
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model_path = self.get_model_path(model_filename)
        self.model = self.load_model(self.model_path)

    def get_model_path(self, model_filename):
        """
        Constructs the absolute path to the model file.
        Ensures the file exists before proceeding.
        """
        model_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "Models", model_filename)
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        return model_path

    def load_model(self, model_path):
        """
        Loads the VGG11 model with batch normalization and applies the trained weights.
        """
        # Use VGG11 with batch normalization
        model = models.vgg11_bn(weights=None)  # Do not use pre-trained weights
        model.classifier[6] = torch.nn.Linear(
            4096, 4
        )  # Adjust final layer to 4 classes

        # Load weights with strict=False to handle mismatches
        model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True),
            strict=False,
        )
        model.to(self.device)
        model.eval()
        return model

    def transform_image(self, image_path):
        """
        Preprocesses an image for model inference.
        """
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        image = Image.open(image_path).convert("RGB")
        return transform(image).unsqueeze(0).to(self.device)

    def predict(self, image_path):
        """
        Runs inference on an image and returns detailed prediction information.
        """
        # Define a mapping of class indices to names
        class_names = {0: "Ampolla", 1: "Mancha", 2: "Pústula", 3: "Roncha"}

        image_tensor = self.transform_image(image_path)
        with torch.no_grad():
            output = self.model(image_tensor)  # Raw model outputs (logits)
            probabilities = F.softmax(output, dim=1).squeeze(0)  # Convert logits to probabilities and remove batch dimension
            prediction = torch.argmax(output, dim=1).item()  # Predicted class index

        # Map the prediction index to the class name
        prediction_name = class_names.get(prediction, "Unknown")
        prediction_percentage = f"{probabilities[prediction].item() * 100:.2f}%"  # Get the probability of the predicted class

        # Include all predictions with their probabilities
        all_predictions = [
            {"class": class_names.get(idx, "Unknown"), "percentage": f"{prob.item() * 100:.2f}%"}
            for idx, prob in enumerate(probabilities)
        ]

        # Return the desired JSON structure
        return {
            "real_prediction": [
                {"percentage": prediction_percentage, "prediction": prediction_name}
            ],
            "all_predictions": all_predictions
        }
