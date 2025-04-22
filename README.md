# Dermatoss Backend

This is a Flask-based backend application for image classification and inference using a pre-trained VGG11 model. It also integrates with MongoDB for storing and retrieving related data.

## Features

- Image classification with a pre-trained VGG11 model.
- REST API endpoints for inference and health checks.
- MongoDB integration for storing and retrieving related images and data.


## Prerequisites

Before running the application, ensure you have the following installed:

- Python 3.12 or higher
- pip (Python package manager)
- MongoDB instance (local or cloud-based)
- [Git](https://git-scm.com/) (optional, for cloning the repository)

## Installation Guide

Follow these steps to set up and run the application locally:

### 1. Clone the Repository

```bash
git clone https://github.com/ArmandoArV/TC3002-Backend.git
cd dermatoss-backend
```

### 2. Set Up a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
Install the required Python packages listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory (if not already present) and configure the following variables:
```.env
MONGO_URI=<your-mongodb-uri>
MONGO_DB_NAME=<your-database-name>
```
Replace `<your-mongodb-uri>` and `<your-database-name>` with your MongoDB connection details.

### 5. Run the Application
Start the Flask application:
```bash
python main.py
```

The application will run on `http://localhost:5000` by default.

### 6. Test the Endpoints
- Health Check: Visit `http://localhost:5000/healthcheck/health` to check the application's health status.
- Inference: Use a tool like Postman or curl to send a POST request to `http://localhost:5000/inference/predict` with an image file.
Example `curl` Command for Inference
```bash
curl -X POST -F "image=@path/to/your/image.jpg" http://localhost:5000/inference/predict
```

## Docker Support
You can also run the application using Docker:

### 1. Build the Docker Image
```bash
docker build -t dermatoss-backend .
```
### 2. Run the Docker Container
```bash
docker run -p 5000:5000 --env-file .env dermatoss-backend
```
The application will be accessible at `http://localhost:5000`.
## Project Structure
```
.
├── .env                  # Environment variables
├── Dockerfile            # Docker configuration
├── main.py               # Entry point for the application
├── requirements.txt      # Python dependencies
├── src/                  # Source code
│   ├── app.py            # Flask app setup
│   ├── Controllers/      # Business logic
│   ├── Database/         # MongoDB connection
│   ├── Models/           # Pre-trained model files
│   ├── Routes/           # API routes
│   └── info.json         # Metadata for predictions
└── uploads/              # Directory for uploaded files
```