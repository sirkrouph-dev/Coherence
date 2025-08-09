# Tutorial: Getting Started with Neuromorphic Programming

A step-by-step guide to using the Neuromorphic Programming System.

## Table of Contents

1. [Installation](#installation)
2. [Basic Network Creation](#basic-network-creation)
3. [Neuron Models](#neuron-models)
4. [Synaptic Plasticity](#synaptic-plasticity)
5. [Sensory Encoding](#sensory-encoding)
6. [Neuromodulation](#neuromodulation)
7. [Advanced Features](#advanced-features)
8. [Jetson Deployment](#jetson-deployment)

## Installation

### Prerequisites

- Python 3.8+
- NumPy, SciPy, Matplotlib
- (Optional) CuPy for GPU acceleration

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd neuron

# Install dependencies
pip install -r requirements.txt

# Test installation
python tests/test_system.py
```

## Basic Network Creation

### Step 1: Create a Simple Network

```python
from api.neuromorphic_api import NeuromorphicAPI

# Initialize the API
api = NeuromorphicAPI()
api.create_network()

# Add layers
api.add_sensory_layer("input", 50, "rate")
api.add_processing_layer("hidden", 25, "adex")
api.add_motor_layer("output", 10)

# Connect layers
api.connect_layers("input", "hidden", "feedforward", synapse_type="stdp")
api.connect_layers("hidden", "output", "feedforward", synapse_type="stdp")

# Run simulation
results = api.run_simulation(100.0)

# Print results
print(f"Simulation completed in {results['duration']} ms")
for layer_name, spikes in results['layer_spike_times'].items():
    print(f"{layer_name}: {len(spikes)} spikes")
```

### Step 2: Add External Input

```python
# Create input spikes
input_spikes = [(i, i * 10.0) for i in range(20)]

# Run simulation with external input
results = api.run_simulation(100.0, external_inputs={"input": input_spikes})

# Visualize results
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
for i, (layer_name, spikes) in enumerate(results['layer_spike_times'].items()):
    plt.subplot(len(results['layer_spike_times']), 1, i+1)
    if spikes:
        times = [spike[1] for spike in spikes]
        neurons = [spike[0] for spike in spikes]
        plt.scatter(times, neurons, alpha=0.6, s=10)
    plt.title(f'{layer_name} Layer Spikes')
    plt.ylabel('Neuron ID')
plt.xlabel('Time (ms)')
plt.tight_layout()
plt.show()
```

## Neuron Models

### Understanding Neuron Types

The system supports three neuron models:

1. **AdEx (Adaptive Exponential Integrate-and-Fire)**: Most biologically realistic
2. **HH (Hodgkin-Huxley)**: Full ion channel dynamics
3. **LIF (Leaky Integrate-and-Fire)**: Simple and efficient

### Creating Individual Neurons

```python
from core.neurons import AdaptiveExponentialIntegrateAndFire, LeakyIntegrateAndFire

# Create an AdEx neuron
adex_neuron = AdaptiveExponentialIntegrateAndFire(
    neuron_id=0,
    v_rest=-65.0,
    v_thresh=-50.0,
    v_reset=-65.0,
    tau_m=20.0,
    a=1.0,
    b=0.0,
    delta_t=2.0
)

# Create a LIF neuron
lif_neuron = LeakyIntegrateAndFire(
    neuron_id=1,
    v_rest=-65.0,
    v_thresh=-50.0,
    v_reset=-65.0,
    tau_m=20.0
)

# Simulate neurons
import numpy as np

# AdEx neuron simulation
adex_spikes = []
for t in range(1000):
    current = 1.0 if 100 < t < 200 else 0.0  # Input current
    if adex_neuron.step(dt=0.1, input_current=current):
        adex_spikes.append(t * 0.1)

# LIF neuron simulation
lif_spikes = []
for t in range(1000):
    current = 1.0 if 100 < t < 200 else 0.0
    if lif_neuron.step(dt=0.1, input_current=current):
        lif_spikes.append(t * 0.1)

print(f"AdEx spikes: {len(adex_spikes)}")
print(f"LIF spikes: {len(lif_spikes)}")
```

## Synaptic Plasticity

### STDP Learning

Spike-Timing-Dependent Plasticity allows synapses to strengthen or weaken based on spike timing.

```python
from core.synapses import STDP_Synapse, SynapseType

# Create STDP synapse
synapse = STDP_Synapse(
    synapse_id=0,
    pre_neuron_id=0,
    post_neuron_id=1,
    weight=1.0,
    synapse_type=SynapseType.EXCITATORY,
    tau_stdp=20.0,
    A_plus=0.01,
    A_minus=0.01
)

# Simulate STDP learning
print(f"Initial weight: {synapse.weight}")

# Pre-synaptic spike before post-synaptic (LTP)
synapse.pre_spike(t=10.0)
synapse.post_spike(t=12.0)
print(f"Weight after LTP: {synapse.weight}")

# Post-synaptic spike before pre-synaptic (LTD)
synapse.pre_spike(t=20.0)
synapse.post_spike(t=18.0)
print(f"Weight after LTD: {synapse.weight}")
```

### Reward-Modulated STDP

Combines timing-based learning with reward signals.

```python
from core.synapses import RSTDP_Synapse

# Create RSTDP synapse
rstdp_synapse = RSTDP_Synapse(
    synapse_id=0,
    pre_neuron_id=0,
    post_neuron_id=1,
    weight=1.0,
    learning_rate=0.01
)

# Set neuromodulator level and reward
rstdp_synapse.update_neuromodulator(level=0.5)
rstdp_synapse.update_reward(reward=1.0)

# Simulate learning
print(f"Initial weight: {rstdp_synapse.weight}")
rstdp_synapse.pre_spike(t=10.0)
rstdp_synapse.post_spike(t=12.0)
print(f"Weight after RSTDP: {rstdp_synapse.weight}")
```

## Sensory Encoding

### Visual Encoding

Convert images to spike trains using retinal-like processing.

```python
from core.encoding import RetinalEncoder
import numpy as np

# Create visual encoder
encoder = RetinalEncoder(resolution=(32, 32))

# Create a simple image (circle in center)
image = np.zeros((32, 32))
center_x, center_y = 16, 16
radius = 8

for y in range(32):
    for x in range(32):
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        if distance <= radius:
            image[y, x] = 1.0

# Encode image to spikes
spikes = encoder.encode(image)
print(f"Generated {len(spikes)} visual spikes")

# Visualize spike pattern
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

# Original image
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Spike raster
plt.subplot(1, 3, 2)
if spikes:
    times = [spike[1] for spike in spikes]
    neurons = [spike[0] for spike in spikes]
    plt.scatter(times, neurons, alpha=0.6, s=10)
plt.title('Spike Raster')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron ID')

# Spike histogram
plt.subplot(1, 3, 3)
if spikes:
    times = [spike[1] for spike in spikes]
    plt.hist(times, bins=20, alpha=0.7)
plt.title('Spike Histogram')
plt.xlabel('Time (ms)')
plt.ylabel('Count')

plt.tight_layout()
plt.show()
```

### Auditory Encoding

Convert audio signals to spike trains using cochlear-like processing.

```python
from core.encoding import CochlearEncoder

# Create auditory encoder
encoder = CochlearEncoder(frequency_bands=64)

# Create a simple audio signal (tone)
sample_rate = 44100
duration = 0.1  # 100ms
t = np.linspace(0, duration, int(sample_rate * duration))
frequency = 440.0  # A4 note
audio = np.sin(2 * np.pi * frequency * t)

# Encode audio to spikes
spikes = encoder.encode(audio)
print(f"Generated {len(spikes)} auditory spikes")
```

### Multi-Modal Encoding

Combine multiple sensory modalities.

```python
from core.encoding import MultiModalEncoder, RetinalEncoder, CochlearEncoder, SomatosensoryEncoder

# Create multi-modal encoder
encoder = MultiModalEncoder({
    'vision': RetinalEncoder(resolution=(32, 32)),
    'auditory': CochlearEncoder(frequency_bands=64),
    'tactile': SomatosensoryEncoder(sensor_grid=(16, 16))
})

# Create multi-modal input
inputs = {
    'vision': np.random.rand(32, 32),
    'auditory': np.random.randn(4410),  # 100ms at 44.1kHz
    'tactile': np.random.rand(16, 16)
}

# Encode multi-modal input
spikes = encoder.encode(inputs)
print(f"Generated {len(spikes)} multi-modal spikes")
```

## Neuromodulation

### Understanding Neuromodulatory Systems

The system includes four major neuromodulatory systems:

1. **Dopamine**: Reward prediction and learning
2. **Serotonin**: Mood and behavioral state
3. **Acetylcholine**: Attention and learning rate
4. **Norepinephrine**: Arousal and vigilance

### Using Neuromodulatory Controllers

```python
from core.neuromodulation import NeuromodulatoryController, NeuromodulatorType

# Create neuromodulatory controller
controller = NeuromodulatoryController()

# Update neuromodulatory systems
controller.update(
    sensory_input=np.array([1.0]),
    reward=1.0,
    expected_reward=0.5,
    positive_events=1,
    negative_events=0,
    threat_signals=0.0,
    task_difficulty=0.5,
    dt=0.1
)

# Get neuromodulator levels
levels = controller.get_modulator_levels()
print(f"Dopamine: {levels[NeuromodulatorType.DOPAMINE]:.3f}")
print(f"Serotonin: {levels[NeuromodulatorType.SEROTONIN]:.3f}")
print(f"Acetylcholine: {levels[NeuromodulatorType.ACETYLCHOLINE]:.3f}")
print(f"Norepinephrine: {levels[NeuromodulatorType.NOREPINEPHRINE]:.3f}")
```

### Adaptive Learning

```python
from core.neuromodulation import AdaptiveLearningController

# Create adaptive learning controller
adaptive_controller = AdaptiveLearningController()

# Update learning rates based on network state
network_info = {
    'connections': {
        'input->hidden': {'synapses': 100},
        'hidden->output': {'synapses': 50}
    }
}

learning_rate = adaptive_controller.update_learning_rates(network_info)
print(f"Adaptive learning rate: {learning_rate:.4f}")
```

## Advanced Features

### Sensorimotor Learning

Create a complete sensorimotor learning system.

```python
from api.neuromorphic_api import SensorimotorSystem

# Create sensorimotor system
system = SensorimotorSystem()

# Create training data
training_data = []
for i in range(20):
    training_data.append({
        'visual': np.random.rand(32, 32),
        'auditory': np.random.randn(4410),
        'tactile': np.random.rand(16, 16),
        'target': i % 8,  # 8 motor neurons
        'reward': 1.0 if i % 2 == 0 else -0.2
    })

# Train the system
training_results = system.train(training_data, epochs=10)

# Test the trained system
test_inputs = {
    'visual': np.random.rand(32, 32),
    'auditory': np.random.randn(4410),
    'tactile': np.random.rand(16, 16)
}

results = system.run_trial(test_inputs)
print(f"Test completed with {len(results['layer_spike_times'])} layers")
```

### Network Visualization

```python
from api.neuromorphic_api import NeuromorphicVisualizer

# Create visualizer
visualizer = NeuromorphicVisualizer()

# Create network for visualization
api = NeuromorphicAPI()
api.create_network()
api.add_sensory_layer("input", 20, "rate")
api.add_processing_layer("hidden", 10, "adex")
api.add_motor_layer("output", 5)
api.connect_layers("input", "hidden", "feedforward")
api.connect_layers("hidden", "output", "feedforward")

# Visualize network
visualizer.plot_network_structure(api.network)
visualizer.plot_weight_matrices(api.network)
```

## Jetson Deployment

### Basic Jetson Setup

```python
from jetson_optimization import JetsonOptimizer, JetsonSensorimotorSystem

# Create Jetson optimizer
optimizer = JetsonOptimizer()

# Get system information
system_info = optimizer.get_system_info()
print(f"CPU cores: {system_info['cpu_count']}")
print(f"Available memory: {system_info['memory_available'] / (1024**3):.2f} GB")
print(f"Temperature: {system_info['temperature']:.1f}°C")

# Create Jetson-optimized system
jetson_system = JetsonSensorimotorSystem(use_gpu=True)
jetson_system.initialize()

# Run inference
inputs = {
    'vision': np.random.rand(16, 16),
    'auditory': np.random.randn(100),
    'tactile': np.random.rand(8, 8)
}

results = jetson_system.run_inference(inputs, duration=50.0)

# Get performance metrics
metrics = results.get('jetson_metrics', {})
print(f"CPU usage: {metrics.get('current_cpu', 0):.1f}%")
print(f"Memory usage: {metrics.get('current_memory', 0):.1f}%")
print(f"Temperature: {metrics.get('current_temperature', 0):.1f}°C")
print(f"Power consumption: {metrics.get('current_power', 0):.2f}W")
```

### Performance Monitoring

```python
import time

# Monitor performance over time
for i in range(10):
    start_time = time.time()
    
    # Run inference
    results = jetson_system.run_inference(inputs, duration=10.0)
    
    inference_time = time.time() - start_time
    metrics = results.get('jetson_metrics', {})
    
    print(f"Trial {i+1}:")
    print(f"  Inference time: {inference_time:.3f}s")
    print(f"  CPU: {metrics.get('current_cpu', 0):.1f}%")
    print(f"  Memory: {metrics.get('current_memory', 0):.1f}%")
    print(f"  Temperature: {metrics.get('current_temperature', 0):.1f}°C")
    print()
    
    time.sleep(1)
```

## Best Practices

### 1. Network Design

- Start with small networks and scale up
- Use appropriate neuron models for your application
- Balance biological realism with computational efficiency

### 2. Learning Configuration

- Adjust STDP parameters based on your task
- Monitor learning convergence
- Use neuromodulation for complex behaviors

### 3. Performance Optimization

- Use LIF neurons for large networks
- Reduce connection probability for efficiency
- Monitor memory usage on resource-constrained systems

### 4. Jetson Deployment

- Monitor temperature and power consumption
- Reduce network size for edge deployment
- Use GPU acceleration when available

## Common Issues and Solutions

### Memory Issues

```python
# Reduce network size
api.add_sensory_layer("input", 25, "rate")  # Reduced from 50
api.add_processing_layer("hidden", 12, "lif")  # Use LIF instead of AdEx

# Clear memory
import gc
gc.collect()
```

### Learning Convergence

```python
# Adjust STDP parameters
api.connect_layers("input", "hidden", "feedforward", 
                  synapse_type="stdp",
                  A_plus=0.005,  # Reduced from 0.01
                  A_minus=0.005,
                  tau_stdp=30.0)  # Increased from 20.0
```

### Jetson Performance

```python
# Check system resources
system_info = optimizer.get_system_info()
if system_info['temperature'] > 70:
    print("Warning: High temperature detected")
    # Reduce workload or increase cooling

if system_info['memory_available'] < 1 * (1024**3):  # Less than 1GB
    print("Warning: Low memory")
    # Reduce network size
```

## Next Steps

1. **Explore the demos**: Run `python demo/sensorimotor_demo.py`
2. **Experiment with parameters**: Try different neuron and synapse configurations
3. **Build custom applications**: Use the API to create your own neuromorphic systems
4. **Deploy to Jetson**: Follow the Jetson deployment guide
5. **Contribute**: Add new features or improve existing ones

For more information, see the [API Reference](API_REFERENCE.md) and [Jetson Deployment Guide](../JETSON_DEPLOYMENT.md). 