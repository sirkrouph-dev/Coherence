# Docker Support for Neuromorphic System

This document provides comprehensive instructions for using Docker with the Neuromorphic System project.

## Table of Contents
- [Quick Start](#quick-start)
- [Available Images](#available-images)
- [Docker Compose Services](#docker-compose-services)
- [Development Workflow](#development-workflow)
- [Production Deployment](#production-deployment)
- [GPU Support](#gpu-support)
- [Troubleshooting](#troubleshooting)

## Quick Start

### Prerequisites
- Docker Engine 20.10+
- Docker Compose 2.0+
- (Optional) NVIDIA Docker for GPU support
- (Optional) Make for convenient commands

### Basic Usage

1. **Build and start all services:**
   ```bash
   make docker-build
   make docker-up
   ```

2. **Start Jupyter Lab:**
   ```bash
   make docker-jupyter
   # Access at http://localhost:8888
   # Default token: neuron2024
   ```

3. **Run tests:**
   ```bash
   make docker-test
   ```

4. **Run a demo:**
   ```bash
   make docker-demo
   ```

## Available Images

The project provides three Docker image variants:

### 1. Runtime Image (`neuron:latest`)
- Minimal production-ready image
- Contains only runtime dependencies
- Optimized for size and security
- Use for production deployments

```bash
docker run ghcr.io/neuromorphic-system/neuron:latest
```

### 2. Development Image (`neuron:dev`)
- Includes Jupyter Lab and development tools
- Contains testing and linting utilities
- Ideal for interactive development

```bash
docker run -p 8888:8888 ghcr.io/neuromorphic-system/neuron:dev
```

### 3. GPU Image (`neuron:gpu`)
- Includes CUDA and GPU libraries
- Requires NVIDIA Docker runtime
- For GPU-accelerated workloads

```bash
docker run --gpus all ghcr.io/neuromorphic-system/neuron:gpu
```

## Docker Compose Services

The `docker-compose.yml` file defines multiple services:

### Core Services

- **neuron**: Main runtime service
- **jupyter**: Jupyter Lab for interactive development
- **test**: Automated test runner
- **lint**: Code quality checks

### Demo Services

- **demo-sensorimotor**: Sensorimotor integration demo
- **demo-training**: Training pipeline demo
- **demo-pattern**: Pattern completion demo
- **demo-sequence**: Sequence learning demo
- **neuron-gpu**: GPU-accelerated demos

### Utility Services

- **benchmark**: Performance benchmarking
- **lint**: Code formatting and linting

### Running Services

```bash
# Start a specific service
docker-compose up jupyter

# Run a one-off command
docker-compose run --rm test

# View service logs
docker-compose logs -f jupyter

# Stop all services
docker-compose down
```

## Development Workflow

### 1. Initial Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/neuromorphic-system.git
cd neuromorphic-system

# Build development image
make docker-build-dev

# Start Jupyter Lab
make docker-jupyter
```

### 2. Interactive Development

The development image includes Jupyter Lab with all dependencies:

```bash
# Start Jupyter with mounted volumes
docker-compose up -d jupyter

# Open shell in container
make docker-shell

# Run Python REPL
docker-compose run --rm jupyter python
```

### 3. Testing

```bash
# Run all tests
make docker-test

# Run specific test file
docker-compose run --rm test pytest tests/test_neurons.py

# Run with coverage
docker-compose run --rm test pytest --cov=core --cov-report=html
```

### 4. Code Quality

```bash
# Run all linters
make docker-lint

# Format code
docker-compose run --rm lint black .
docker-compose run --rm lint isort .
```

## Production Deployment

### Building for Production

```bash
# Build optimized image
DOCKER_BUILDKIT=1 docker build \
  --target runtime \
  --tag ghcr.io/neuromorphic-system/neuron:latest \
  .

# Multi-platform build
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --target runtime \
  --tag ghcr.io/neuromorphic-system/neuron:latest \
  --push \
  .
```

### Deploying to Kubernetes

Example Kubernetes deployment:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: neuron-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: neuron
  template:
    metadata:
      labels:
        app: neuron
    spec:
      containers:
      - name: neuron
        image: ghcr.io/neuromorphic-system/neuron:latest
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
```

### Docker Swarm Deployment

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml neuron-stack

# Scale service
docker service scale neuron-stack_neuron=5
```

## GPU Support

### Prerequisites

1. NVIDIA GPU with CUDA support
2. NVIDIA Docker runtime installed
3. CUDA 11.8+ drivers

### Setup

```bash
# Verify GPU availability
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Build GPU image
make docker-build-gpu

# Run GPU demo
docker-compose run --rm neuron-gpu
```

### GPU Docker Compose

```yaml
services:
  neuron-gpu:
    image: ghcr.io/neuromorphic-system/neuron:gpu
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Environment Variables

Configure services using environment variables:

```bash
# Create .env file
cat > .env << EOF
JUPYTER_TOKEN=your-secure-token
JUPYTER_PORT=8888
GITHUB_REPOSITORY=yourusername/neuromorphic-system
PYTHONUNBUFFERED=1
EOF

# Start services with custom config
docker-compose up
```

## Volumes and Data Persistence

The Docker setup uses named volumes for data persistence:

- `jupyter_data`: Jupyter configuration and notebooks
- `test_cache`: Test cache and results
- `./data`: Input data directory
- `./outputs`: Output results directory
- `./models`: Trained models directory

### Backup and Restore

```bash
# Backup volumes
docker run --rm \
  -v neuron_jupyter_data:/data \
  -v $(pwd)/backup:/backup \
  alpine tar czf /backup/jupyter_data.tar.gz -C /data .

# Restore volumes
docker run --rm \
  -v neuron_jupyter_data:/data \
  -v $(pwd)/backup:/backup \
  alpine tar xzf /backup/jupyter_data.tar.gz -C /data
```

## CI/CD with GitHub Actions

The project includes automated Docker builds via GitHub Actions:

1. **Automatic builds** on push to main/develop branches
2. **Multi-platform support** (amd64, arm64)
3. **Vulnerability scanning** with Trivy
4. **Automatic push** to GitHub Container Registry

### Manual Deployment

```bash
# Login to GitHub Container Registry
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin

# Build and push
make docker-build
make docker-push
```

## Troubleshooting

### Common Issues

1. **Permission denied errors:**
   ```bash
   # Add user to docker group
   sudo usermod -aG docker $USER
   # Log out and back in
   ```

2. **Out of memory errors:**
   ```bash
   # Increase Docker memory limit
   # Docker Desktop: Settings -> Resources -> Memory
   ```

3. **GPU not detected:**
   ```bash
   # Verify NVIDIA Docker installation
   docker run --rm --gpus all nvidia/cuda:11.8.0-base nvidia-smi
   ```

4. **Jupyter connection refused:**
   ```bash
   # Check if port is already in use
   lsof -i :8888
   # Use different port
   JUPYTER_PORT=8889 docker-compose up jupyter
   ```

### Debugging

```bash
# Interactive debugging
docker-compose run --rm jupyter bash
python -m pdb demo/sensorimotor_demo.py

# View container logs
docker-compose logs -f --tail=100 jupyter

# Inspect running container
docker exec -it neuron-jupyter bash

# Check resource usage
docker stats
```

## Security Considerations

1. **Non-root user**: Containers run as non-root user `neuron`
2. **Minimal base image**: Using Python slim images
3. **Multi-stage builds**: Separate build and runtime stages
4. **Secret management**: Use Docker secrets or environment variables
5. **Image scanning**: Regular vulnerability scans with Trivy

### Security Scanning

```bash
# Scan image for vulnerabilities
trivy image ghcr.io/neuromorphic-system/neuron:latest

# Generate detailed report
trivy image --format json --output trivy-report.json \
  ghcr.io/neuromorphic-system/neuron:latest
```

## Performance Optimization

### Image Size Optimization

- Multi-stage builds reduce final image size by ~60%
- Slim base images (python:3.11-slim)
- Minimal runtime dependencies
- Cached pip wheels

### Build Performance

```bash
# Enable BuildKit for faster builds
export DOCKER_BUILDKIT=1

# Use build cache
docker build --cache-from ghcr.io/neuromorphic-system/neuron:latest .

# Parallel builds
docker-compose build --parallel
```

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
- [GitHub Container Registry](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)

## Support

For issues or questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review [GitHub Issues](https://github.com/yourusername/neuromorphic-system/issues)
3. Create a new issue with the `docker` label
