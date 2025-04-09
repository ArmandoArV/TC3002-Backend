from flask import Blueprint, request, jsonify
import os
from werkzeug.utils import secure_filename
from src.Controllers.Inference_Controller import InferenceController

inference_bp = Blueprint('inference', __name__)
controller = InferenceController()
UPLOAD_FOLDER = "uploads"

# Ensure the upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@inference_bp.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']

    if image_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded file
    filename = secure_filename(image_file.filename)
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    image_file.save(image_path)

    # Run the inference
    result = controller.predict(image_path)

    # Optionally delete the file after processing
    os.remove(image_path)

    return jsonify(result)