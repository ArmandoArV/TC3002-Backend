import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F  # Import for softmax
import os
import json
from pymongo import MongoClient  # Import MongoDB client
from src.Database.mongo_connection import MongoConnection
import unicodedata

# Import the VGG model and related functions
from src.Models.vgg import VGG, get_vgg_layers, vgg16_config

class InferenceController:
    def __init__(self, model_filename="best_model.pt", device=None):
        """
        Initialize the inference controller.
        - Loads the model from the correct path.
        - Uses the available device (GPU or CPU).
        - Reuses the MongoDB connection.
        """
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model_path = self.get_model_path(model_filename)
        self.model = self.load_model(self.model_path)

        # Reuse the MongoDB connection
        mongo_instance = MongoConnection.get_instance()
        self.db = mongo_instance.db

    def get_related_images(self, prediction_name):
        """
        Retrieves all related images from MongoDB based on the prediction name.
        Dynamically selects the collection based on the prediction.
        """
        # Normalize the prediction name to remove accents
        normalized_prediction_name = unicodedata.normalize('NFKD', prediction_name).encode('ASCII', 'ignore').decode('utf-8')
        collection_name = f"{normalized_prediction_name.lower()}"
        print(f"Collection name: {collection_name}")

        # Check if the collection exists
        if collection_name not in self.db.list_collection_names():
            print(f"Collection '{collection_name}' does not exist.")
            return []

        # Access the collection dynamically
        collection = self.db[collection_name]
        print(f"Connected to MongoDB database: {self.db.name}")
        print(f"Number of documents in collection '{collection_name}': {collection.count_documents({})}")

        # Query the collection for all images, filtering out documents without the 'image' field
        related_images = collection.find({}, {"_id": 0, "image": 1})
        images = [doc["image"] for doc in related_images if "image" in doc]
        print(f"Retrieved images: {images}")

        return images

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
        Loads the VGG16 model and applies the trained weights.
        """
        # Use VGG16
        model = VGG(get_vgg_layers(vgg16_config, batch_norm=True), output_dim=4)  # Adjust the output dimension to 4 for your classes

        # Load weights with strict=False to handle mismatches
        model.load_state_dict(
            torch.load(model_path, map_location=self.device),
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
            output, _ = self.model(image_tensor)  # Raw model outputs (logits)
            probabilities = F.softmax(output, dim=1).squeeze(0)  # Convert logits to probabilities and remove batch dimension
            prediction = torch.argmax(output, dim=1).item()  # Predicted class index

        # Map the prediction index to the class name
        prediction_name = class_names.get(prediction, "Unknown")
        prediction_percentage = f"{probabilities[prediction].item() * 100:.2f}%"  # Get the probability of the predicted class

        # Normalize the prediction name to use it as collection name
        normalized_prediction_name = unicodedata.normalize('NFKD', prediction_name).encode('ASCII', 'ignore').decode('utf-8')
        collection_name = f"{normalized_prediction_name.lower()}"

        # Retrieve the 'info' document from MongoDB
        info_doc = self.db[collection_name].find_one({"type": "info"}, {"_id": 0}) or {}

        # Retrieve all related images
        related_images = self.get_related_images(prediction_name)

        # Include all predictions with their probabilities
        all_predictions = [
            {"class": class_names.get(idx, "Unknown"), "percentage": f"{prob.item() * 100:.2f}%"}
            for idx, prob in enumerate(probabilities)
        ]

        # Return the desired JSON structure
        return {
            "real_prediction": [
                {
                    "percentage": prediction_percentage,
                    "prediction": prediction_name,
                    "info_elemental": info_doc.get("info_elemental", ""),
                    "caracteristicas_clave": info_doc.get("caracteristicas_clave", ""),
                    "mas_informacion": info_doc.get("mas_informacion", ""),
                    "url": info_doc.get("url", ""),
                    "related_images": related_images
                }
            ],
            "all_predictions": all_predictions
        }
