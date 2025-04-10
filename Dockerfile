# Use an official Python runtime as a parent image
FROM python:3.10.11

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . /app/

# Expose the port your application runs on (adjust if necessary)
EXPOSE 8000

# Set the command to run your application
# Replace 'main.py' with the entry point of your application
CMD ["python", "main.py"]