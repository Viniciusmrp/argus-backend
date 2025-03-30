# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies needed for OpenCV
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --default-timeout=120 -r /app/requirements.txt

# Copy the rest of the application into the container at /app
COPY . /app

# Expose the port the app runs on
EXPOSE 8080

# Run the Flask app
CMD ["python", "app.py"]
