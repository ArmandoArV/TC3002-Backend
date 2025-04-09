import torch
from torchvision import models, transforms
from PIL import Image
import os

class InferenceController:
    def __init__(self, model_path="models/vgg11_model.pth", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        model = models.vgg11(pretrained=False)  # Load model architecture
        model.classifier[6] = torch.nn.Linear(4096, 10)  # Adjust final layer
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def transform_image(self, image_path):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = Image.open(image_path).convert("RGB")
        return transform(image).unsqueeze(0).to(self.device)

    def predict(self, image_path):
        image_tensor = self.transform_image(image_path)
        with torch.no_grad():
            output = self.model(image_tensor)
        prediction = torch.argmax(output, dim=1).item()
        return {"prediction": prediction}
