from flask import Flask
from flask_cors import CORS
from src.Routes.Inference import inference_bp


app = Flask(__name__)

CORS(app)

app.register_blueprint(inference_bp, url_prefix='/inference')



