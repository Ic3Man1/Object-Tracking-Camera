# Use ARM32/ARM64-compatible Python base image for Raspberry Pi 4
FROM python:3.9-slim-bullseye

# Set environment variables for ARM optimization
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV DOCKER_DEFAULT_PLATFORM=linux/arm/v7

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV and YOLO on ARM
RUN apt-get update && \
    apt-get install -y \
        # OpenCV dependencies
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        libgstreamer1.0-0 \
        libgstreamer-plugins-base1.0-0 \
        libgtk-3-0 \
        libavcodec58 \
        libavformat58 \
        libswscale5 \
        # Build tools for compiling packages
        build-essential \
        cmake \
        pkg-config \
        # Additional libraries for ARM optimization
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libdc1394-22-dev \
        libv4l-dev \
        # Cleanup
        && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python packages with ARM32-friendly approach
RUN pip install --no-cache-dir --upgrade pip && \
    # Install PyTorch CPU version (fallback to CPU-only if ARM32 wheels unavailable)
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu || \
    pip install --no-cache-dir torch torchvision --extra-index-url https://pypi.org/simple/ && \
    # Try pre-built OpenCV first, fallback to system package
    pip install --no-cache-dir opencv-python-headless || \
    (apt-get update && apt-get install -y python3-opencv && \
     ln -sf /usr/lib/python3/dist-packages/cv2 /usr/local/lib/python3.9/site-packages/) && \
    # Install ultralytics with fallback
    pip install --no-cache-dir ultralytics || \
    pip install --no-cache-dir ultralytics --no-deps && \
    # Install other requirements
    pip install --no-cache-dir -r requirements.txt || true && \
    # Clean up
    pip cache purge && rm -rf /var/lib/apt/lists/*

# Copy application files
COPY . .

# Create directory for model weights if it doesn't exist
RUN mkdir -p /app/models

# Set proper permissions
RUN chmod +x /app/*.py

# Health check (optional)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import cv2, torch, ultralytics; print('Dependencies OK')" || exit 1

# Expose port if your application needs it
# EXPOSE 8000

# Command to run the application
CMD ["python", "yolo.py"]