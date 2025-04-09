from flask import Flask
from flask_cors import CORS

app = Flask(__name__)

CORS(app)

@app.route('/api/health', methods=['GET'])

def health_check():
    return {'status': 'ok'}, 200




