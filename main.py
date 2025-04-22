from src.app import app
import os

# Use environment variables for configuration
HOST = "localhost"# os.getenv('FLASK_RUN_HOST', '0.0.0.0')
PORT = 5000 # int(os.getenv('FLASK_RUN_PORT', 5000))
DEBUG = False # os.getenv('FLASK_DEBUG', 'False').lower() in ['true', '1', 'yes']



if __name__ == '__main__':
    app.run(host=HOST, port=PORT, debug=DEBUG)