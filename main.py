from src.app import app
import os

# Use environment variables for configuration
HOST = os.getenv('FLASK_RUN_HOST', '0.0.0.0')
PORT = int(os.getenv('FLASK_RUN_PORT', 5000))
DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() in ['true', '1', 'yes']

if __name__ == '__main__':
    app.run(host=HOST, port=PORT, debug=DEBUG)