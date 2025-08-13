# Multi-stage build for neuromorphic-system
# Stage 1: Builder stage with full dependencies
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /build

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    cmake \
    libopencv-dev \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml requirements.txt ./

# Install Python build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Create wheels for dependencies
RUN pip wheel --no-cache-dir --wheel-dir /build/wheels -r requirements.txt

# Stage 2: Runtime stage with minimal footprint
FROM python:3.11-slim as runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 neuron && \
    mkdir -p /app && \
    chown -R neuron:neuron /app

# Set working directory
WORKDIR /app

# Copy wheels from builder stage
COPY --from=builder /build/wheels /wheels

# Install dependencies from wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/* && \
    rm -rf /wheels

# Copy application code
COPY --chown=neuron:neuron . .

# Install the package in development mode
RUN pip install --no-cache-dir -e .

# Switch to non-root user
USER neuron

# Default command
CMD ["python", "-m", "demo.sensorimotor_demo"]

# Stage 3: Development stage with Jupyter and dev tools
FROM runtime as development

# Switch back to root for installation
USER root

# Install development dependencies
RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    ipython \
    ipywidgets \
    pytest \
    pytest-cov \
    pytest-benchmark \
    black \
    ruff \
    isort \
    mypy

# Create jupyter config directory
RUN mkdir -p /home/neuron/.jupyter && \
    chown -R neuron:neuron /home/neuron/.jupyter

# Switch back to non-root user
USER neuron

# Expose Jupyter port
EXPOSE 8888

# Default command for development
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# Stage 4: GPU-enabled runtime (optional)
FROM runtime as gpu

# Switch to root for CUDA installation
USER root

# Install CUDA dependencies (adjust version as needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-minimal-build-11-8 \
    libcudnn8 \
    && rm -rf /var/lib/apt/lists/*

# Install GPU-specific Python packages
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    cupy-cuda11x \
    nvidia-ml-py

# Switch back to non-root user
USER neuron

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
