# Use official Python runtime as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the worker script
COPY worker.py .

# Create temp directory for file operations
RUN mkdir -p /tmp

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose port (optional, for debugging)
EXPOSE 8080

# Run the worker
CMD ["python", "worker.py"]
