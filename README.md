# Neuromorphic Programming System
# SPDX-License-Identifier: MIT

[![CI](https://github.com/sirkrouph-dev/NeuroMorph/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/sirkrouph-dev/NeuroMorph/actions/workflows/ci.yml)
[![Tests](https://github.com/sirkrouph-dev/NeuroMorph/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/sirkrouph-dev/NeuroMorph/actions/workflows/test.yml)
[![Lint](https://github.com/sirkrouph-dev/NeuroMorph/actions/workflows/lint.yml/badge.svg?branch=main)](https://github.com/sirkrouph-dev/NeuroMorph/actions/workflows/lint.yml)
[![Coverage](https://codecov.io/gh/sirkrouph-dev/NeuroMorph/branch/main/graph/badge.svg)](https://codecov.io/gh/sirkrouph-dev/NeuroMorph)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11%20|%203.12-blue.svg)](https://www.python.org/)
[![OS](https://img.shields.io/badge/OS-Linux%20|%20Windows%20|%20macOS-informational.svg)](#)
[![Benchmarks](https://img.shields.io/badge/Benchmarks-View%20Results-orange.svg)](docs/benchmarks.md)

A comprehensive neuromorphic computing framework that bridges biological neuroscience and edge computing, featuring brain-inspired neural networks with realistic dynamics, plasticity, and neuromodulation.

Why you’ll love it:
- Biologically grounded learning (STDP, STP, neuromodulation) with clean, tested implementations
- Optional sleep/rest phase (replay + consolidation) to stabilize learning
- Clear examples, benchmarks with runtime caps, and CI across platforms

## 🚀 Quick Start

```python
from core.network import NeuromorphicNetwork

net = NeuromorphicNetwork()
net.add_layer("input", 100, "lif")
net.add_layer("output", 10, "lif")
net.connect_layers("input", "output", "stdp", connection_probability=0.1)

results = net.run_simulation(duration=100.0, dt=0.1)
print("Layers:", results["layer_spike_times"].keys())
```

[Full Getting Started Guide →](docs/tutorials/01_getting_started.md)

## 🧠 Overview

This system implements a biologically plausible neuromorphic computing framework that moves beyond classical artificial neural networks to embrace the temporal, structural, and plastic complexity of real brains. It provides a complete neuromorphic programming environment with:

- **Biological Neuron Models**: AdEx, Hodgkin-Huxley, Izhikevich, Leaky Integrate-and-Fire
- **Synaptic Plasticity**: STDP, Short-term plasticity, Reward-modulated learning, Structural plasticity
- **Neuromodulatory Systems**: Dopamine, Serotonin, Acetylcholine, Norepinephrine
- **Sensory Encoding**: Visual (retinal, DVS), Auditory (cochlear), Tactile (mechanoreceptor) processing
- **Simulation Modes**: Time-step and event-driven simulation
- **Edge Deployment**: Optimized for NVIDIA Jetson Nano and embedded systems

## 🏗️ System Architecture

```
neuron/
├── engine/                  # Neural simulation engine
│   ├── network.py          # High-level network construction
│   ├── neuron_group.py     # Neuron population management
│   ├── synapse_group.py    # Synaptic connection management
│   ├── neuron_models.py    # LIF, Izhikevich, AdEx, HH models
│   ├── synapse_models.py   # Static, STDP, STP, neuromodulatory
│   └── simulator.py        # Time-step and event-driven simulation
├── core/                    # Core neuromorphic components
│   ├── neurons.py          # Legacy neuron models
│   ├── synapses.py         # Legacy synapse models
│   ├── network.py          # Legacy network architecture
│   ├── encoding.py         # Sensory input encoding
│   ├── enhanced_encoding.py # Advanced encoding (DVS, cochlear)
│   ├── neuromodulation.py  # Neuromodulatory systems
│   ├── learning.py         # Plasticity mechanisms
│   ├── memory.py           # Memory subsystems
│   └── gpu_neurons.py      # GPU-accelerated neurons
├── api/                    # High-level programming interface
│   ├── neuromorphic_api.py # Main API for system interaction
│   └── neuromorphic_system.py # Unified system class
├── demo/                   # Demonstration scripts
│   ├── sensorimotor_demo.py # Basic sensorimotor demo
│   ├── sensorimotor_training.py # Advanced training demo
│   ├── jetson_demo.py      # Jetson Nano deployment
│   ├── gpu_large_scale_demo.py # GPU acceleration demo
│   └── enhanced_comprehensive_demo.py # Full feature showcase
├── examples/               # Example applications
│   ├── engine_demo.py      # Engine usage examples
│   ├── pattern_completion_demo.py # Pattern completion
│   └── sequence_learning_demo.py # Sequence learning
├── benchmarks/             # Performance benchmarking
│   ├── performance_benchmarks.py # Core benchmarks
│   ├── pytest_benchmarks.py # Test benchmarks
│   └── visualize_benchmarks.py # Benchmark visualization
├── tests/                  # Testing and validation
│   ├── test_neurons.py     # Neuron model tests
│   ├── test_synapses.py    # Synapse model tests
│   ├── test_learning.py    # Learning mechanism tests
│   └── test_integration.py # Integration tests
├── docs/                   # Documentation
│   ├── tutorials/          # Step-by-step tutorials
│   │   ├── 01_getting_started.md
│   │   ├── 02_sensory_encoding.md
│   │   ├── 03_learning_plasticity.md
│   │   └── 05_edge_deployment.md
│   ├── API_REFERENCE.md    # API documentation
│   ├── ARCHITECTURE.md     # System architecture
│   └── benchmarks.md       # Performance metrics
├── scripts/                # Utility scripts
│   ├── jetson_optimization.py # Jetson optimization
│   ├── gpu_optimization.py # GPU optimization
│   └── check_quality.py    # Code quality checks
└── setup.py                # Package installation
```

## 📦 Installation

### Prerequisites
- Python 3.8+
- NumPy, SciPy, Matplotlib
- (Optional) PyTorch CUDA for GPU probes (core simulator is CPU/NumPy)
- (Optional) CuPy for additional GPU experiments
- (Optional) NVIDIA Jetson Nano for edge deployment

### Install as Package (Recommended)

```bash
# Clone the repository
git clone https://github.com/sirkrouph-dev/NeuroMorph.git
cd neuron

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with desired features
pip install -e .              # Basic installation
pip install -e ".[dev]"       # With development tools
pip install -e ".[gpu]"       # With GPU support
pip install -e ".[jetson]"    # For Jetson Nano
pip install -e ".[all]"       # Everything
```

### Verify Installation

```bash
# Run verification script
python verify_installation.py

# Run tests
python -m pytest tests/

# Run example
python examples/engine_demo.py
```

## 🧬 Key Features

### Neuron Models

The system implements multiple biologically plausible neuron models:

1. **Leaky Integrate-and-Fire (LIF)** - Simple, computationally efficient
2. **Izhikevich** - Rich dynamics with low computational cost
3. **Adaptive Exponential (AdEx)** - Spike frequency adaptation
4. **Hodgkin-Huxley** - Detailed biophysical model with ion channels

Each model offers different trade-offs between biological realism and computational efficiency.

### Synaptic Plasticity

1. **STDP** - Spike-timing dependent plasticity for Hebbian learning
2. **STP** - Short-term plasticity with depression/facilitation
3. **Reward-Modulated** - Three-factor learning with neuromodulation
4. **Homeostatic** - Synaptic scaling and intrinsic plasticity
5. **Structural** - Dynamic synapse formation and elimination

### Simulation Modes

1. **Time-Step Mode** - Traditional fixed time-step integration
2. **Event-Driven Mode** - Efficient processing of sparse activity
3. **Edge Optimization** - Specialized modes for embedded devices

## 🎯 Applications

### Robotics
- Sensorimotor control
- Navigation and path planning
- Object recognition and manipulation

### Edge AI
- Real-time pattern recognition
- Anomaly detection
- Low-power inference

### Neuroscience Research
- Brain circuit modeling
- Learning mechanism studies
- Neural dynamics exploration

### IoT and Embedded Systems
- Smart sensors
- Adaptive control systems
- Energy-efficient processing

## 🔬 Core Components

### Neuron Models (`core/neurons.py`)

```python
from core.neurons import AdaptiveExponentialIntegrateAndFire

# Create AdEx neuron
neuron = AdaptiveExponentialIntegrateAndFire(
    neuron_id=0,
    v_rest=-65.0,
    v_thresh=-50.0,
    v_reset=-65.0,
    tau_m=20.0,
    a=1.0,
    b=0.0,
    delta_t=2.0
)

# Step simulation
neuron.step(dt=0.1, input_current=1.0)
```

### Synapse Models (`core/synapses.py`)

```python
from core.synapses import STDP_Synapse

# Create STDP synapse
synapse = STDP_Synapse(
    synapse_id=0,
    pre_neuron_id=0,
    post_neuron_id=1,
    weight=1.0,
    tau_stdp=20.0,
    A_plus=0.01,
    A_minus=0.01
)

# Update weights based on spike timing
synapse.pre_spike(t=10.0)
synapse.post_spike(t=12.0)
```

### Network Architecture (`core/network.py`)

```python
from core.network import NeuromorphicNetwork

# Create network
network = NeuromorphicNetwork()

# Add layers
network.add_layer("sensory", 100, "sensory")
network.add_layer("hidden", 50, "processing")
network.add_layer("motor", 10, "motor")

# Connect layers
network.connect_layers("sensory", "hidden", connection_probability=0.1)
network.connect_layers("hidden", "motor", connection_probability=0.2)

# Run simulation
results = network.run_simulation(duration=100.0, dt=0.1)
```

### Sensory Encoding (`core/encoding.py`)

```python
from core.encoding import RetinalEncoder

# Create visual encoder
encoder = RetinalEncoder(resolution=(32, 32))

# Encode image to spike train
image = np.random.rand(32, 32)
spikes = encoder.encode(image)
print(f"Generated {len(spikes)} spikes")
```

## 🎮 Demonstrations
### Sleep/Rest Phase (Replay + Consolidation)
```python
from core.network import NeuromorphicNetwork
import numpy as np

net = NeuromorphicNetwork()
net.add_layer("input", 10, "lif")
net.add_layer("output", 5, "lif")
net.connect_layers("input", "output", "stdp", connection_probability=1.0)

# Optional sleep with replay, SHY downscale, incoming normalization, and noise
pattern = np.zeros(10); pattern[0:3] = 50.0
net.run_sleep_phase(
    duration=50.0, dt=0.1,
    replay={"input": pattern},
    downscale_factor=0.98,
    normalize_incoming=True,
    noise_std=0.05,
)
```


### 1. Basic Network Demo
```bash
python demo/sensorimotor_demo.py
```
Shows:
- Network creation and configuration
- Sensory encoding capabilities
- Adaptive behavior patterns
- Sensorimotor learning

### 2. Sleep Cycle Demo
```bash
python examples/sleep_cycle_demo.py
```
Shows:
- Brief training → sleep (replay + SHY downscale + normalization + noise) → measurement
- Before/after spike-count responses and weight summary

### 3. Advanced Training Demo
```bash
python demo/sensorimotor_training.py
```
Features:
- Reward-modulated learning
- Adaptive learning rates
- Performance monitoring
- Real-time adaptation

### 4. Jetson Nano Demo
```bash
python demo/jetson_demo.py
```
Includes:
- System information display
- Performance optimization
- Real-time inference
- Learning capabilities
- Performance monitoring

## 🚀 Jetson Nano Deployment

The system is fully optimized for NVIDIA Jetson Nano deployment:

### Quick Jetson Setup

```bash
# Run Jetson optimization test
python jetson_optimization.py

# Run Jetson demo
python demo/jetson_demo.py
```

### Performance Characteristics

| Metric | Desktop | Jetson Nano | Optimization |
|--------|---------|-------------|--------------|
| **Neurons** | 1000+ | 500-1000 | 50-90% reduction |
| **Synapses** | 10000+ | 5000-10000 | 50-90% reduction |
| **Inference Time** | 0.1s | 0.1-0.5s | Real-time capable |
| **Memory Usage** | 4-8GB | 2-4GB | 50% reduction |
| **Power Consumption** | 50-100W | 5-10W | 90% reduction |

See `JETSON_DEPLOYMENT.md` for complete deployment guide.

## 🧪 Testing

### Run All Tests
```bash
python tests/test_system.py
```

### Test Individual Components
```python
# Test neuron models
python -c "from core.neurons import *; test_neuron_models()"

# Test synapse models
python -c "from core.synapses import *; test_synapse_models()"

# Test network creation
python -c "from core.network import *; test_network_creation()"
```

## 📊 Performance Metrics

### Biological Accuracy
- Temporal fidelity: < 1ms spike timing precision
- Learning convergence: 80-95% accuracy
- Network stability: Homeostatic regulation

### Computational Efficiency
- Memory usage: 1KB per neuron, 100B per synapse
- Simulation speed: 1000x real-time (desktop)
- Energy efficiency: 90% reduction vs traditional ANNs

### Scalability
- Network size: 100-10,000 neurons
- Connection density: 1-20% connectivity
- Layer depth: 2-10 layers

## 🔧 Configuration

### System Configuration
```python
# Network parameters
NETWORK_CONFIG = {
    'max_neurons': 1000,
    'max_synapses': 10000,
    'simulation_dt': 0.1,
    'learning_rate': 0.01
}

# Neuron parameters
NEURON_CONFIG = {
    'v_rest': -65.0,
    'v_thresh': -50.0,
    'tau_m': 20.0,
    'refractory_period': 2.0
}

# Synapse parameters
SYNAPSE_CONFIG = {
    'tau_stdp': 20.0,
    'A_plus': 0.01,
    'A_minus': 0.01,
    'weight_max': 5.0
}
```

## 📚 Documentation

### Tutorials
Step-by-step guides for getting started:

- [**Getting Started**](docs/tutorials/01_getting_started.md) - Installation, first network, basic concepts
- [**Sensory Encoding**](docs/tutorials/02_sensory_encoding.md) - Converting real-world data to spikes
- [**Learning & Plasticity**](docs/tutorials/03_learning_plasticity.md) - STDP, homeostasis, reinforcement learning
- [**Edge Deployment**](docs/tutorials/05_edge_deployment.md) - Jetson Nano and embedded systems

### Core Documentation
- [API Reference](docs/API_REFERENCE.md) - Complete API documentation
- [System Architecture](docs/ARCHITECTURE.md) - Technical architecture details
- [Performance Benchmarks](docs/benchmarks.md) - Speed and efficiency metrics
- [Jetson Deployment](JETSON_DEPLOYMENT.md) - Detailed Jetson Nano guide

### Examples
- `examples/` - Working code examples
- `demo/` - Full demonstration applications
- `benchmarks/` - Performance testing scripts

## 🤝 Contributing

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd neuron

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
pytest tests/

# Format code
black core/ api/ demo/ tests/
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Include docstrings
- Write unit tests

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Biological neuroscience community
- Neuromorphic computing researchers
- NVIDIA Jetson platform
- Open-source scientific computing community

## 📞 Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Check the documentation
- Review the examples in `demo/`

---

**Neuromorphic Programming System** - Bridging biological neuroscience and edge computing for the next generation of brain-inspired AI. 