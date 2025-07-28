# Use AMD64-compatible Python base image
FROM --platform=linux/amd64 python:3.10-slim

# Set working directory
WORKDIR /app

# Copy everything except virtualenv and __pycache__
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Offline-compatible: avoid internet at runtime
# Entry point: run main.py which handles everything
CMD ["python", "main.py"]
