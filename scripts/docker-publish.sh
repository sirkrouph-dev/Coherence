#!/bin/bash
# Script to build and publish Docker images to GitHub Container Registry

set -e

# Configuration
REGISTRY="ghcr.io"
NAMESPACE="${GITHUB_REPOSITORY:-neuromorphic-system}"
IMAGE_NAME="${REGISTRY}/${NAMESPACE}/neuron"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed or not in PATH"
    exit 1
fi

# Check if logged in to GitHub Container Registry
if ! docker info 2>/dev/null | grep -q "${REGISTRY}"; then
    print_warning "Not logged in to ${REGISTRY}"
    print_status "Please run: echo \$GITHUB_TOKEN | docker login ${REGISTRY} -u USERNAME --password-stdin"
    exit 1
fi

# Parse command line arguments
BUILD_TARGETS=("runtime" "development" "gpu")
PUSH_IMAGES=false
TAG_VERSION=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --push)
            PUSH_IMAGES=true
            shift
            ;;
        --tag)
            TAG_VERSION="$2"
            shift 2
            ;;
        --target)
            BUILD_TARGETS=("$2")
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --push          Push images to registry after building"
            echo "  --tag VERSION   Tag images with specific version"
            echo "  --target TARGET Build specific target (runtime|development|gpu)"
            echo "  --help          Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Get git commit hash and branch
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
GIT_TAG=$(git describe --tags --exact-match 2>/dev/null || echo "")

print_status "Building Docker images..."
print_status "Repository: ${NAMESPACE}"
print_status "Commit: ${GIT_COMMIT}"
print_status "Branch: ${GIT_BRANCH}"
if [ -n "${GIT_TAG}" ]; then
    print_status "Tag: ${GIT_TAG}"
fi

# Enable Docker BuildKit
export DOCKER_BUILDKIT=1

# Build each target
for TARGET in "${BUILD_TARGETS[@]}"; do
    print_status "Building ${TARGET} image..."
    
    # Determine tag suffix
    case ${TARGET} in
        runtime)
            TAG_SUFFIX="latest"
            ;;
        development)
            TAG_SUFFIX="dev"
            ;;
        gpu)
            TAG_SUFFIX="gpu"
            ;;
        *)
            TAG_SUFFIX="${TARGET}"
            ;;
    esac
    
    # Build image
    docker build \
        --target "${TARGET}" \
        --tag "${IMAGE_NAME}:${TAG_SUFFIX}" \
        --tag "${IMAGE_NAME}:${GIT_COMMIT}-${TAG_SUFFIX}" \
        --tag "${IMAGE_NAME}:${GIT_BRANCH}-${TAG_SUFFIX}" \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        --cache-from "${IMAGE_NAME}:${TAG_SUFFIX}" \
        .
    
    # Add version tag if specified
    if [ -n "${TAG_VERSION}" ]; then
        docker tag "${IMAGE_NAME}:${TAG_SUFFIX}" "${IMAGE_NAME}:${TAG_VERSION}-${TAG_SUFFIX}"
        print_status "Tagged as ${IMAGE_NAME}:${TAG_VERSION}-${TAG_SUFFIX}"
    fi
    
    # Add git tag if exists
    if [ -n "${GIT_TAG}" ]; then
        docker tag "${IMAGE_NAME}:${TAG_SUFFIX}" "${IMAGE_NAME}:${GIT_TAG}-${TAG_SUFFIX}"
        print_status "Tagged as ${IMAGE_NAME}:${GIT_TAG}-${TAG_SUFFIX}"
    fi
    
    print_status "Successfully built ${TARGET} image"
done

# Show built images
print_status "Built images:"
docker images "${IMAGE_NAME}" --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"

# Push images if requested
if [ "${PUSH_IMAGES}" = true ]; then
    print_status "Pushing images to ${REGISTRY}..."
    
    for TARGET in "${BUILD_TARGETS[@]}"; do
        case ${TARGET} in
            runtime)
                TAG_SUFFIX="latest"
                ;;
            development)
                TAG_SUFFIX="dev"
                ;;
            gpu)
                TAG_SUFFIX="gpu"
                ;;
            *)
                TAG_SUFFIX="${TARGET}"
                ;;
        esac
        
        # Push all tags
        docker push "${IMAGE_NAME}:${TAG_SUFFIX}"
        docker push "${IMAGE_NAME}:${GIT_COMMIT}-${TAG_SUFFIX}"
        docker push "${IMAGE_NAME}:${GIT_BRANCH}-${TAG_SUFFIX}"
        
        if [ -n "${TAG_VERSION}" ]; then
            docker push "${IMAGE_NAME}:${TAG_VERSION}-${TAG_SUFFIX}"
        fi
        
        if [ -n "${GIT_TAG}" ]; then
            docker push "${IMAGE_NAME}:${GIT_TAG}-${TAG_SUFFIX}"
        fi
        
        print_status "Pushed ${TARGET} images"
    done
    
    print_status "All images pushed successfully!"
else
    print_status "Images built but not pushed. Use --push to push to registry."
fi

# Scan for vulnerabilities (optional)
if command -v trivy &> /dev/null; then
    print_status "Running security scan with Trivy..."
    trivy image --severity HIGH,CRITICAL "${IMAGE_NAME}:latest"
else
    print_warning "Trivy not installed. Skipping security scan."
    print_warning "Install from: https://github.com/aquasecurity/trivy"
fi

print_status "Done!"
