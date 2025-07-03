# Use Python image with system packages
FROM python:3.10-slim

# Install system-level packages needed for librosa, soundfile, pyworld, etc.
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    build-essential \
    libgl1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy code
COPY . /app

# Install PyTorch (CPU-only wheel)
RUN pip install --upgrade pip
RUN pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Install rest of the dependencies
RUN pip install -r requirements.txt

# Expose port
EXPOSE 5000

# Start server
CMD ["python", "server.py"]
