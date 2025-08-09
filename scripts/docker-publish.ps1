# PowerShell script to build and publish Docker images to GitHub Container Registry

param(
    [switch]$Push,
    [string]$Tag = "",
    [string]$Target = "",
    [switch]$Help
)

# Configuration
$Registry = "ghcr.io"
$Namespace = if ($env:GITHUB_REPOSITORY) { $env:GITHUB_REPOSITORY } else { "neuromorphic-system" }
$ImageName = "$Registry/$Namespace/neuron"

# Colors for output
function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] " -ForegroundColor Green -NoNewline
    Write-Host $Message
}

function Write-Error-Message {
    param([string]$Message)
    Write-Host "[ERROR] " -ForegroundColor Red -NoNewline
    Write-Host $Message
}

function Write-Warning-Message {
    param([string]$Message)
    Write-Host "[WARNING] " -ForegroundColor Yellow -NoNewline
    Write-Host $Message
}

# Show help
if ($Help) {
    Write-Host "Usage: .\docker-publish.ps1 [options]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Push           Push images to registry after building"
    Write-Host "  -Tag VERSION    Tag images with specific version"
    Write-Host "  -Target TARGET  Build specific target (runtime|development|gpu)"
    Write-Host "  -Help           Show this help message"
    exit 0
}

# Check if Docker is available
try {
    docker version | Out-Null
} catch {
    Write-Error-Message "Docker is not installed or not running"
    exit 1
}

# Build targets
$BuildTargets = if ($Target) { @($Target) } else { @("runtime", "development", "gpu") }

# Get git information
$GitCommit = & git rev-parse --short HEAD 2>$null
if (-not $GitCommit) { $GitCommit = "unknown" }

$GitBranch = & git rev-parse --abbrev-ref HEAD 2>$null
if (-not $GitBranch) { $GitBranch = "unknown" }

$GitTag = & git describe --tags --exact-match 2>$null

Write-Status "Building Docker images..."
Write-Status "Repository: $Namespace"
Write-Status "Commit: $GitCommit"
Write-Status "Branch: $GitBranch"
if ($GitTag) {
    Write-Status "Tag: $GitTag"
}

# Enable Docker BuildKit
$env:DOCKER_BUILDKIT = "1"

# Build each target
foreach ($BuildTarget in $BuildTargets) {
    Write-Status "Building $BuildTarget image..."
    
    # Determine tag suffix
    $TagSuffix = switch ($BuildTarget) {
        "runtime" { "latest" }
        "development" { "dev" }
        "gpu" { "gpu" }
        default { $BuildTarget }
    }
    
    # Build image
    $BuildArgs = @(
        "build",
        "--target", $BuildTarget,
        "--tag", "${ImageName}:${TagSuffix}",
        "--tag", "${ImageName}:${GitCommit}-${TagSuffix}",
        "--tag", "${ImageName}:${GitBranch}-${TagSuffix}",
        "--build-arg", "BUILDKIT_INLINE_CACHE=1",
        "--cache-from", "${ImageName}:${TagSuffix}",
        "."
    )
    
    $Result = & docker $BuildArgs 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Error-Message "Failed to build $BuildTarget image"
        Write-Host $Result
        exit 1
    }
    
    # Add version tag if specified
    if ($Tag) {
        docker tag "${ImageName}:${TagSuffix}" "${ImageName}:${Tag}-${TagSuffix}"
        Write-Status "Tagged as ${ImageName}:${Tag}-${TagSuffix}"
    }
    
    # Add git tag if exists
    if ($GitTag) {
        docker tag "${ImageName}:${TagSuffix}" "${ImageName}:${GitTag}-${TagSuffix}"
        Write-Status "Tagged as ${ImageName}:${GitTag}-${TagSuffix}"
    }
    
    Write-Status "Successfully built $BuildTarget image"
}

# Show built images
Write-Status "Built images:"
docker images $ImageName --format "table {{.Repository}}:{{.Tag}}`t{{.Size}}`t{{.CreatedAt}}"

# Push images if requested
if ($Push) {
    Write-Status "Pushing images to $Registry..."
    
    foreach ($BuildTarget in $BuildTargets) {
        $TagSuffix = switch ($BuildTarget) {
            "runtime" { "latest" }
            "development" { "dev" }
            "gpu" { "gpu" }
            default { $BuildTarget }
        }
        
        # Push all tags
        docker push "${ImageName}:${TagSuffix}"
        docker push "${ImageName}:${GitCommit}-${TagSuffix}"
        docker push "${ImageName}:${GitBranch}-${TagSuffix}"
        
        if ($Tag) {
            docker push "${ImageName}:${Tag}-${TagSuffix}"
        }
        
        if ($GitTag) {
            docker push "${ImageName}:${GitTag}-${TagSuffix}"
        }
        
        Write-Status "Pushed $BuildTarget images"
    }
    
    Write-Status "All images pushed successfully!"
} else {
    Write-Status "Images built but not pushed. Use -Push to push to registry."
}

# Scan for vulnerabilities (optional)
$TrivyExists = Get-Command trivy -ErrorAction SilentlyContinue
if ($TrivyExists) {
    Write-Status "Running security scan with Trivy..."
    trivy image --severity HIGH,CRITICAL "${ImageName}:latest"
} else {
    Write-Warning-Message "Trivy not installed. Skipping security scan."
    Write-Warning-Message "Install from: https://github.com/aquasecurity/trivy"
}

Write-Status "Done!"
