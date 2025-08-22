# Neuromorphic Computing System - AI Coding Instructions

## System Overview & Vision
This project is evolving from a core neuromorphic framework into a **comprehensive, interactive playground for brain simulation**. The primary goal is to create an easy-to-use, web-based platform where developers (including non-neuroscientists) can build, visualize, and experiment with biologically realistic spiking neural networks.

The long-term vision includes a community platform for sharing models and fostering collaborative research. The project is structured in phases, starting with core enhancements, then adding cognitive functions, building the interactive web UI, and finally launching a community platform.

## Architecture & Component Boundaries

### Core Simulation Engine (`core/`)
- **`balanced_competitive_learning.py`**: The foundational algorithm for stable concept representation. It's a key feature but now part of a larger system.
- **`enhanced_neurons.py` / `neurons.py`**: Implementation of diverse, biologically plausible neuron models (e.g., Regular Spiking, Fast Spiking, Bursting).
- **`enhanced_synapses.py` / `synapses.py`**: Manages multi-plasticity rules (STDP, homeostatic, metaplasticity) and neuromodulation.
- **`brain_topology.py` / `network.py`**: The `NeuromorphicNetwork` class orchestrates simulations. The `BrainTopologyBuilder` will create realistic network structures (e.g., distance-dependent connectivity, E/I balance).
- **Cognitive Components**:
    - `sensory_hierarchy.py`: For hierarchical sensory processing.
    - `working_memory.py`: For implementing working memory via persistent activity.
    - `neural_oscillations.py`: For analyzing emergent brain rhythms like Gamma and Theta.

### High-Level API (`api/`)
- **`neuromorphic_api.py`**: A simplified API for programmatic network construction. This will be the backend for the web playground.

### Interactive Web Playground (`web/`)
- **Backend (`web/backend.py`)**: A **FastAPI** application that serves the simulation engine via WebSockets.
- **Frontend (`web/frontend/`)**: A **React** application with **D3.js** for real-time visualization of neural activity.

## Development Workflows

### Essential Commands
```bash
# Basic setup
pip install -e .

# Setup with optional features
pip install -e ".[dev]"     # Development tools
pip install -e ".[gpu]"     # GPU support (CuPy/PyTorch)
pip install -e ".[jetson]"  # Jetson optimization
pip install -e ".[all]"     # Install everything

# Testing
python -m pytest tests/                    # Run all unit tests
python -m pytest tests/test_integration.py # Run integration tests
python benchmarks/performance_benchmarks.py # Run performance benchmarks

# Code Quality
black core/ api/ web/ tests/   # Code formatting
ruff check .                    # Linting
mypy core/ api/ web/            # Type checking
```

### Key Demos & Experiments
The ultimate demo is the web playground itself. Key scripts for testing core functionality remain important.
```bash
# High-level capability demos
python tools/agi_testbed_complete.py    # Full framework demonstration

# Core functionality demos
python demo/sensorimotor_demo.py        # Basic sensorimotor learning loop
python experiments/learning_assessment.py  # Framework evaluation suite
```

## Project-Specific Patterns

### Building a Realistic Network
The new primary pattern is using the `BrainTopologyBuilder` to create complex, biologically-inspired networks.
```python
from core.brain_topology import BrainTopologyBuilder

builder = BrainTopologyBuilder()
# Creates a network with modules, E/I balance, and distance-based connections
network = builder.create_cortical_network(size=1000, modules=4) 
results = network.run_simulation(duration=100.0, dt=0.1)
```

### Implementing a New Neuron Model
- Inherit from `NeuronModel` in `core/neurons.py`.
- Implement the `step(dt, I_syn)` method, which should return `True` if the neuron spikes.
- Add presets for your new neuron type to the `NeuronFactory`.

### Web Playground Interaction
- The frontend communicates with the backend via **WebSockets** for real-time data streaming (`/ws/neural_activity`).
- Network configurations are sent to the backend via REST API endpoints (e.g., `POST /create_network`).

## Critical Dependencies
- **NumPy/SciPy**: For all numerical computations.
- **CuPy/PyTorch**: For optional GPU acceleration.
- **FastAPI**: For the web backend.
- **React & D3.js**: For the web frontend.
- **OpenCV**: Used in `encoding.py` for image processing.
- **psutil**: For monitoring system resources.

## Common Gotchas & Constraints
- **Real-time Simulation**: The web playground requires the simulation to run in near real-time. This necessitates performance optimization (vectorization, sparse matrices, adaptive time steps).
- **Scalability**: The system is designed to scale from small networks on CPU to large networks (50k+ neurons) on GPU.
- **Biological Plausibility**: While the goal is an accessible playground, the underlying models should adhere to plausible biological principles and parameter ranges.
- **Asynchronous Communication**: The web UI and the simulation engine run asynchronously. Proper handling of state and communication is critical.
