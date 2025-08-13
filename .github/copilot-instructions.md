# Neuromorphic Computing System - AI Coding Instructions

## System Overview
This is a **biologically-inspired neuromorphic computing framework** implementing spiking neural networks with temporal dynamics, synaptic plasticity (STDP), and neuromodulation. The system bridges neuroscience research and edge computing with deployment optimization for NVIDIA Jetson and GPU acceleration.

## Architecture & Component Boundaries

### Core Simulation Engine (`core/`)
- **`neurons.py`**: Biologically plausible models (AdEx, HH, LIF, Izhikevich) inheriting from `NeuronModel` base class
- **`synapses.py`**: Synaptic transmission and plasticity (`STDP_Synapse`, `STP_Synapse`) with weight updates
- **`network.py`**: `NeuromorphicNetwork` class managing layers via `NetworkLayer` objects and event-driven simulation
- **`encoding.py`**: Sensory input conversion (`RetinalEncoder`, `CochlearEncoder`, `SomatosensoryEncoder`)
- **`neuromodulation.py`**: Dopamine/serotonin systems affecting learning and homeostasis

### High-Level API (`api/`)
- **`neuromorphic_api.py`**: `NeuromorphicAPI` class providing simplified network creation and simulation
- **`neuromorphic_system.py`**: Unified `SensorimotorSystem` for complete sensorimotor loops

### Platform Optimization
- **GPU acceleration**: Optional CuPy/PyTorch integration with graceful CPU fallback
- **Jetson deployment**: Memory-optimized configurations in `demo/jetson_demo.py`
- **Edge computing**: Real-time inference capabilities with power efficiency

## Development Workflows

### Essential Commands
```bash
# Setup with specific features
pip install -e ".[dev]"     # Development tools
pip install -e ".[gpu]"     # GPU support  
pip install -e ".[jetson]"  # Jetson optimization
pip install -e ".[all]"     # Everything

# Testing hierarchy
python -m pytest tests/                    # Unit tests
python -m pytest tests/test_integration.py # Integration tests
python benchmarks/performance_benchmarks.py # Performance benchmarks

# Quality checks
black core/ api/ demo/ tests/   # Code formatting
ruff check .                    # Linting
mypy core/ api/                 # Type checking

# Run demos
python demo/sensorimotor_demo.py           # Basic sensorimotor learning
python demo/enhanced_comprehensive_demo.py # Full feature showcase
python demo/jetson_demo.py                 # Jetson deployment test
```

### Docker Deployment
Multi-stage builds: `runtime` (production), `development` (Jupyter), `gpu` (CUDA support)
```bash
docker build --target runtime -t neuron:latest .
docker build --target gpu -t neuron:gpu .
```

## Project-Specific Patterns

### Neuron Model Implementation
- Inherit from `NeuronModel` with `step(dt, I_syn)` method returning `bool` (spike/no-spike)
- Use `self.membrane_potential` (alias `self.v`) for voltage state
- Implement `reset()` for initialization between trials
- Store spike times in `self.spike_times` list

### Network Construction Pattern
```python
net = NeuromorphicNetwork()
net.add_layer("input", 100, "lif")          # Size, neuron type
net.add_layer("hidden", 50, "adex")         
net.connect_layers("input", "hidden", "stdp", connection_probability=0.1)
results = net.run_simulation(duration=100.0, dt=0.1)
```

### Sleep/Consolidation Learning
Unique feature: replay-based consolidation with synaptic homeostasis
```python
net.run_sleep_phase(
    duration=50.0, 
    replay={"input": pattern},      # Replay specific patterns
    downscale_factor=0.98,          # Synaptic homeostasis (SHY)
    normalize_incoming=True,        # Weight normalization
    noise_std=0.05                  # Background noise
)
```

### Testing Conventions
- **Unit tests**: Individual neuron/synapse models in `tests/test_neurons.py`, `tests/test_synapses.py`
- **Integration tests**: Full network workflows in `tests/test_integration.py`
- **Performance tests**: Scalability benchmarks with memory profiling in `benchmarks/`
- **GPU tests**: Marked with `@pytest.mark.gpu` for conditional execution

### Platform-Specific Code
- GPU availability detection: `try/except` blocks with graceful CPU fallback
- Jetson optimization: Memory-constrained network sizes (500-1000 neurons max)
- Performance monitoring: `psutil` for memory, timing for benchmarks

## Critical Dependencies & Integration Points

### External Dependencies
- **NumPy/SciPy**: Core numerical computation (all models use numpy arrays)
- **Matplotlib/Seaborn**: Visualization and plotting in API and demos
- **OpenCV**: Image processing for visual encoding
- **CuPy/PyTorch**: Optional GPU acceleration (graceful fallback required)
- **psutil**: System monitoring for benchmarks and Jetson deployment

### Cross-Component Communication
- **Event-driven simulation**: Central event queue in `core.network.EventDrivenSimulator`
- **Neuromodulation**: Controller broadcasts to all synapses via `neuromodulatory_controller`
- **Sensory encoding**: Converts external data to spike trains fed into input layers
- **Logging system**: Centralized in `core.logging_utils` with structured event logging

### Package Structure Transition
**Current state**: Flat structure with direct imports (`from core.neurons import`)
**Production target**: Nested package structure (`src/neuron_sim/`) - see `docs/ARCHITECTURE.md`

## Debugging & Analysis Tools

### Network Analysis
- Spike raster plots: `results["layer_spike_times"]` contains per-layer spike data
- Weight evolution tracking: Access via `synapse.weight` and `synapse.weight_history`
- Performance profiling: Use `benchmarks/performance_benchmarks.py` with memory profiler

### Common Gotchas
- **GPU memory**: Always check CUDA availability before GPU operations
- **Simulation time**: Large networks need runtime caps to prevent infinite execution
- **Weight explosion**: STDP can cause runaway weight growth without homeostatic regulation
- **Platform differences**: Jetson has different CUDA compute capabilities than desktop GPUs

## Biological Accuracy Constraints
- **Temporal precision**: Sub-millisecond spike timing for realistic STDP
- **Parameter ranges**: Biologically plausible values (membrane potentials: -80mV to +40mV)
- **Plasticity rules**: STDP requires pre/post spike timing differences ±100ms window
- **Neuromodulation**: Dopamine affects learning rates, not instant weight changes
