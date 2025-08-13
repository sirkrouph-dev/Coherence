# Getting Started with the Neuromorphic Programming System

A comprehensive guide to installing and using the neuromorphic programming system for the first time.

## Prerequisites

Before you begin, ensure you have:
- Python 3.8 or higher
- pip package manager
- Basic understanding of neural networks
- (Optional) NVIDIA GPU with CUDA support
- (Optional) NVIDIA Jetson Nano for edge deployment

## Installation

### Option 1: Install as Package (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd neuron

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package in editable mode with core dependencies
pip install -e .

# Or install with all development dependencies
pip install -e ".[dev]"

# Or install with GPU support
pip install -e ".[gpu]"

# Or install with Jetson support
pip install -e ".[jetson]"

# Or install everything
pip install -e ".[all]"
```

### Option 2: Install from PyPI (when published)

```bash
# Basic installation
pip install neuromorphic-system

# With extras
pip install neuromorphic-system[gpu]  # For GPU support
pip install neuromorphic-system[dev]  # For development
pip install neuromorphic-system[all]  # Everything
```

### Option 3: Traditional Requirements File

```bash
# Clone the repository
git clone <repository-url>
cd neuron

# Install dependencies
pip install -r requirements.txt
```

## Verifying Installation

After installation, verify everything is working correctly:

```bash
# Run verification script
python verify_installation.py

# Or run tests
python -m pytest tests/

# Or run a simple demo
python demo/sensorimotor_demo.py
```

Expected output:
```
✓ NumPy installed
✓ SciPy installed
✓ Matplotlib installed
✓ Core modules imported successfully
✓ Neuron models working
✓ Synapse models working
✓ Network creation successful
```

## Your First Neuromorphic Network

Let's create a simple feedforward network with three layers:

```python
from api.neuromorphic_api import NeuromorphicAPI

# Initialize the API
api = NeuromorphicAPI()

# Create a new network
api.create_network()

# Add three layers
api.add_sensory_layer("input", neurons=100, encoding_type="rate")
api.add_processing_layer("hidden", neurons=50, neuron_type="adex")
api.add_motor_layer("output", neurons=10)

# Connect layers with plastic synapses
api.connect_layers("input", "hidden", "feedforward", synapse_type="stdp")
api.connect_layers("hidden", "output", "feedforward", synapse_type="stdp")

# Run a simulation for 1000ms
results = api.run_simulation(duration=1000.0)

# Print results
print(f"Simulation completed in {results['duration']} ms")
for layer_name, spikes in results['layer_spike_times'].items():
    print(f"{layer_name}: {len(spikes)} spikes")
```

## Understanding the Components

### Neuron Models

The system provides three biologically plausible neuron models:

1. **AdEx (Adaptive Exponential)**: Most biologically realistic, includes adaptation
2. **LIF (Leaky Integrate-and-Fire)**: Simple and efficient for large-scale simulations
3. **HH (Hodgkin-Huxley)**: Full ion channel dynamics, most computationally expensive

### Synapse Models

Various forms of synaptic plasticity are available:

1. **Static**: Fixed weights, no learning
2. **STDP**: Spike-Timing-Dependent Plasticity for Hebbian learning
3. **STP**: Short-Term Plasticity with depression and facilitation
4. **RSTDP**: Reward-modulated STDP for reinforcement learning

### Network Types

The system supports multiple network architectures:

1. **Feedforward**: Information flows in one direction
2. **Recurrent**: Includes feedback connections
3. **Reservoir**: Random recurrent connections for reservoir computing

## Adding External Input

You can provide external stimulation to your network:

```python
# Create input spikes for specific neurons at specific times
input_spikes = [
    (neuron_id=0, time=10.0),
    (neuron_id=1, time=20.0),
    (neuron_id=2, time=30.0),
]

# Run simulation with external input
results = api.run_simulation(
    duration=1000.0,
    external_inputs={"input": input_spikes}
)
```

## Visualizing Results

Basic visualization of network activity:

```python
import matplotlib.pyplot as plt

# Extract spike data
for layer_name, spikes in results['layer_spike_times'].items():
    if spikes:
        times = [spike[1] for spike in spikes]
        neurons = [spike[0] for spike in spikes]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(times, neurons, alpha=0.6, s=10)
        plt.title(f'{layer_name} Layer Activity')
        plt.xlabel('Time (ms)')
        plt.ylabel('Neuron ID')
        plt.show()
```

## Common Issues and Solutions

### Issue: Import errors
**Solution**: Ensure you're in the correct virtual environment and all dependencies are installed.

### Issue: Memory errors with large networks
**Solution**: Reduce network size or use the Jetson optimization features.

### Issue: Slow simulation
**Solution**: Use LIF neurons instead of HH, or enable GPU acceleration if available.

## Next Steps

Now that you have a working installation and have created your first network, you can:

1. Explore different neuron and synapse models
2. Learn about [sensory encoding](02_sensory_encoding.md)
3. Implement [learning and plasticity](03_learning_plasticity.md)
4. Add [neuromodulation](04_neuromodulation.md)
5. Deploy to [edge devices](05_edge_deployment.md)

## Getting Help

If you encounter issues:
1. Check the [FAQ](../FAQ.md)
2. Review the [API Reference](../API_REFERENCE.md)
3. Search existing GitHub issues
4. Create a new issue with a minimal reproducible example

## Additional Resources

- [System Architecture](../ARCHITECTURE.md)
- [Performance Benchmarks](../benchmarks.md)
- [Contributing Guide](../../CONTRIBUTING.md)
- [Research Papers](../references.md)

---

*Continue to the next tutorial: [Sensory Encoding →](02_sensory_encoding.md)*
