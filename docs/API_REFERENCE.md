# API Reference

Complete API documentation for the Neuromorphic Programming System.

## Table of Contents

1. [Core Components](#core-components)
2. [High-Level API](#high-level-api)
3. [Neuron Models](#neuron-models)
4. [Synapse Models](#synapse-models)
5. [Network Architecture](#network-architecture)
6. [Sensory Encoding](#sensory-encoding)
7. [Neuromodulation](#neuromodulation)
8. [Jetson Optimization](#jetson-optimization)

## Core Components

### NeuromorphicAPI

The main high-level interface for creating and running neuromorphic networks.

```python
from api.neuromorphic_api import NeuromorphicAPI

api = NeuromorphicAPI()
```

#### Methods

##### `create_network()`
Creates a new neuromorphic network.

```python
api.create_network()
```

##### `add_sensory_layer(name, size, encoding_type)`
Adds a sensory input layer.

**Parameters:**
- `name` (str): Layer name
- `size` (int): Number of neurons
- `encoding_type` (str): Encoding type ("rate", "retinal", "cochlear", "somatosensory")

```python
api.add_sensory_layer("visual_input", 64, "retinal")
api.add_sensory_layer("auditory_input", 32, "cochlear")
```

##### `add_processing_layer(name, size, neuron_type)`
Adds a processing layer.

**Parameters:**
- `name` (str): Layer name
- `size` (int): Number of neurons
- `neuron_type` (str): Neuron model ("adex", "hh", "lif")

```python
api.add_processing_layer("hidden", 32, "adex")
```

##### `add_motor_layer(name, size)`
Adds a motor output layer.

**Parameters:**
- `name` (str): Layer name
- `size` (int): Number of neurons

```python
api.add_motor_layer("motor_output", 8)
```

##### `connect_layers(pre_layer, post_layer, connection_type, synapse_type, **kwargs)`
Connects two layers.

**Parameters:**
- `pre_layer` (str): Presynaptic layer name
- `post_layer` (str): Postsynaptic layer name
- `connection_type` (str): Connection pattern ("random", "feedforward", "lateral", "feedback")
- `synapse_type` (str): Synapse model ("stdp", "stp", "neuromodulatory", "rstdp")
- `**kwargs`: Additional synapse parameters

```python
api.connect_layers("input", "hidden", "feedforward", synapse_type="stdp")
```

##### `run_simulation(duration, dt=0.1, external_inputs=None)`
Runs a network simulation.

**Parameters:**
- `duration` (float): Simulation duration in milliseconds
- `dt` (float): Time step in milliseconds
- `external_inputs` (dict): External spike inputs

**Returns:**
- `dict`: Simulation results with spike times and network state

```python
results = api.run_simulation(100.0, external_inputs={"input": [(i, i*10.0) for i in range(20)]})
```

##### `train_sensorimotor_system(training_data, epochs)`
Trains a sensorimotor system.

**Parameters:**
- `training_data` (list): List of training trials
- `epochs` (int): Number of training epochs

**Returns:**
- `dict`: Training results and metrics

```python
training_results = api.train_sensorimotor_system(training_data, epochs=50)
```

##### `get_network_info()`
Gets network information.

**Returns:**
- `dict`: Network statistics and configuration

```python
info = api.get_network_info()
print(f"Network has {info['total_neurons']} neurons")
```

## Neuron Models

### AdaptiveExponentialIntegrateAndFire

Biologically realistic neuron model with spike frequency adaptation.

```python
from core.neurons import AdaptiveExponentialIntegrateAndFire

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
```

#### Parameters

- `neuron_id` (int): Unique neuron identifier
- `v_rest` (float): Resting membrane potential (mV)
- `v_thresh` (float): Spike threshold (mV)
- `v_reset` (float): Reset potential after spike (mV)
- `tau_m` (float): Membrane time constant (ms)
- `a` (float): Subthreshold adaptation parameter
- `b` (float): Spike-triggered adaptation parameter
- `delta_t` (float): Sharpness parameter (mV)

#### Methods

##### `step(dt, input_current)`
Advances neuron state by one time step.

**Parameters:**
- `dt` (float): Time step (ms)
- `input_current` (float): Input current (nA)

**Returns:**
- `bool`: True if neuron spiked

```python
spiked = neuron.step(dt=0.1, input_current=1.0)
if spiked:
    print(f"Neuron {neuron.neuron_id} spiked at time {neuron.last_spike_time}")
```

### HodgkinHuxleyNeuron

Full ion channel dynamics neuron model.

```python
from core.neurons import HodgkinHuxleyNeuron

neuron = HodgkinHuxleyNeuron(
    neuron_id=0,
    v_rest=-65.0,
    g_na=120.0,
    g_k=36.0,
    g_l=0.3
)
```

### LeakyIntegrateAndFire

Simplified but efficient neuron model.

```python
from core.neurons import LeakyIntegrateAndFire

neuron = LeakyIntegrateAndFire(
    neuron_id=0,
    v_rest=-65.0,
    v_thresh=-50.0,
    v_reset=-65.0,
    tau_m=20.0
)
```

## Synapse Models

### STDP_Synapse

Spike-timing-dependent plasticity synapse.

```python
from core.synapses import STDP_Synapse

synapse = STDP_Synapse(
    synapse_id=0,
    pre_neuron_id=0,
    post_neuron_id=1,
    weight=1.0,
    synapse_type=SynapseType.EXCITATORY,
    tau_stdp=20.0,
    A_plus=0.01,
    A_minus=0.01,
    tau_syn=5.0,
    E_rev=0.0
)
```

#### Parameters

- `synapse_id` (int): Unique synapse identifier
- `pre_neuron_id` (int): Presynaptic neuron ID
- `post_neuron_id` (int): Postsynaptic neuron ID
- `weight` (float): Synaptic weight
- `synapse_type` (SynapseType): Excitatory or inhibitory
- `tau_stdp` (float): STDP time constant (ms)
- `A_plus` (float): LTP amplitude
- `A_minus` (float): LTD amplitude
- `tau_syn` (float): Synaptic time constant (ms)
- `E_rev` (float): Reversal potential (mV)

#### Methods

##### `pre_spike(t)`
Handles presynaptic spike.

```python
synapse.pre_spike(t=10.0)
```

##### `post_spike(t)`
Handles postsynaptic spike.

```python
synapse.post_spike(t=12.0)
```

##### `step(dt)`
Advances synapse state.

```python
synapse.step(dt=0.1)
```

### RSTDP_Synapse

Reward-modulated STDP synapse.

```python
from core.synapses import RSTDP_Synapse

synapse = RSTDP_Synapse(
    synapse_id=0,
    pre_neuron_id=0,
    post_neuron_id=1,
    weight=1.0,
    learning_rate=0.01
)
```

#### Methods

##### `update_neuromodulator(level)`
Updates neuromodulator level.

```python
synapse.update_neuromodulator(level=0.5)
```

##### `update_reward(reward)`
Updates reward signal.

```python
synapse.update_reward(reward=1.0)
```

### ShortTermPlasticitySynapse

Synapse with short-term plasticity.

```python
from core.synapses import ShortTermPlasticitySynapse

synapse = ShortTermPlasticitySynapse(
    synapse_id=0,
    pre_neuron_id=0,
    post_neuron_id=1,
    weight=1.0,
    tau_facilitation=100.0,
    tau_depression=200.0,
    U=0.1
)
```

## Network Architecture

### NeuromorphicNetwork

Complete neuromorphic network with layers and connections.

```python
from core.network import NeuromorphicNetwork

network = NeuromorphicNetwork()
```

#### Methods

##### `add_layer(name, size, layer_type)`
Adds a layer to the network.

```python
network.add_layer("sensory", 100, "sensory")
network.add_layer("hidden", 50, "processing")
network.add_layer("motor", 10, "motor")
```

##### `connect_layers(pre_layer, post_layer, connection_probability, **kwargs)`
Connects two layers.

```python
network.connect_layers("sensory", "hidden", connection_probability=0.1)
network.connect_layers("hidden", "motor", connection_probability=0.2)
```

##### `run_simulation(duration, dt)`
Runs network simulation.

```python
results = network.run_simulation(duration=100.0, dt=0.1)
```

### NetworkBuilder

Helper class for building networks.

```python
from core.network import NetworkBuilder

builder = NetworkBuilder()
builder.add_sensory_layer("input", 50, "rate")
builder.add_processing_layer("hidden", 25, "adex")
builder.add_motor_layer("output", 10)
builder.connect_layers("input", "hidden", "feedforward")
builder.connect_layers("hidden", "output", "feedforward")
network = builder.build()
```

## Sensory Encoding

### RetinalEncoder

Visual encoding with retinal-like processing.

```python
from core.encoding import RetinalEncoder

encoder = RetinalEncoder(resolution=(32, 32))
spikes = encoder.encode(image)
```

### CochlearEncoder

Auditory encoding with cochlear-like processing.

```python
from core.encoding import CochlearEncoder

encoder = CochlearEncoder(frequency_bands=64)
spikes = encoder.encode(audio_signal)
```

### SomatosensoryEncoder

Tactile encoding with mechanoreceptor-like processing.

```python
from core.encoding import SomatosensoryEncoder

encoder = SomatosensoryEncoder(sensor_grid=(16, 16))
spikes = encoder.encode(pressure_map)
```

### MultiModalEncoder

Combines multiple sensory modalities.

```python
from core.encoding import MultiModalEncoder

encoder = MultiModalEncoder({
    'vision': RetinalEncoder(resolution=(32, 32)),
    'auditory': CochlearEncoder(frequency_bands=64),
    'tactile': SomatosensoryEncoder(sensor_grid=(16, 16))
})

inputs = {
    'vision': image,
    'auditory': audio,
    'tactile': pressure_map
}
spikes = encoder.encode(inputs)
```

## Neuromodulation

### NeuromodulatoryController

Manages neuromodulatory systems.

```python
from core.neuromodulation import NeuromodulatoryController

controller = NeuromodulatoryController()
```

#### Methods

##### `update(sensory_input, reward, expected_reward, **kwargs)`
Updates neuromodulatory systems.

```python
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
```

##### `get_modulator_levels()`
Gets current neuromodulator levels.

```python
levels = controller.get_modulator_levels()
print(f"Dopamine: {levels[NeuromodulatorType.DOPAMINE]}")
```

### AdaptiveLearningController

Adaptive learning rate controller.

```python
from core.neuromodulation import AdaptiveLearningController

controller = AdaptiveLearningController()
```

#### Methods

##### `update_learning_rates(network_info)`
Updates learning rates based on network state.

```python
controller.update_learning_rates(network_info)
```

##### `apply_learning(network)`
Applies learning to network synapses.

```python
controller.apply_learning(network)
```

## Jetson Optimization

### JetsonOptimizer

Optimizes networks for Jetson Nano deployment.

```python
from jetson_optimization import JetsonOptimizer

optimizer = JetsonOptimizer()
```

#### Methods

##### `get_system_info()`
Gets Jetson system information.

```python
info = optimizer.get_system_info()
print(f"CPU: {info['cpu_count']} cores")
print(f"Memory: {info['memory_available'] / (1024**3):.2f} GB")
print(f"Temperature: {info['temperature']:.1f}°C")
```

##### `optimize_network_size(target_neurons, target_synapses)`
Optimizes network size for Jetson constraints.

```python
optimization = optimizer.optimize_network_size(
    target_neurons=1000,
    target_synapses=10000
)
print(f"Optimized to: {optimization['max_neurons']} neurons")
```

### JetsonSensorimotorSystem

Jetson-optimized sensorimotor system.

```python
from jetson_optimization import JetsonSensorimotorSystem

system = JetsonSensorimotorSystem(use_gpu=True)
system.initialize()
```

#### Methods

##### `run_inference(sensory_inputs, duration)`
Runs inference on Jetson Nano.

```python
inputs = {
    'vision': np.random.rand(16, 16),
    'auditory': np.random.randn(100),
    'tactile': np.random.rand(8, 8)
}
results = system.run_inference(inputs, duration=50.0)
```

##### `get_performance_summary()`
Gets performance summary.

```python
summary = system.get_performance_summary()
print(f"System Info: {summary['system_info']}")
print(f"Network Info: {summary['network_info']}")
```

## Data Structures

### Spike Events

Spike events are represented as tuples: `(neuron_id, spike_time)`

```python
spikes = [(0, 10.0), (1, 12.0), (2, 15.0)]
```

### Network Results

Simulation results contain:

```python
results = {
    'duration': 100.0,
    'layer_spike_times': {
        'input': [(0, 5.0), (1, 8.0)],
        'hidden': [(0, 12.0), (1, 15.0)],
        'output': [(0, 20.0)]
    },
    'weight_matrices': {
        'input->hidden': np.array([[0.5, 0.3], [0.2, 0.8]]),
        'hidden->output': np.array([[0.6, 0.4]])
    }
}
```

### Training Data

Training data format:

```python
training_data = [
    {
        'vision': np.random.rand(32, 32),
        'auditory': np.random.randn(44100),
        'tactile': np.random.rand(16, 16),
        'target': 0,
        'reward': 1.0
    }
]
```

## Error Handling

### Common Exceptions

```python
try:
    api.run_simulation(100.0)
except ValueError as e:
    print(f"Invalid parameter: {e}")
except RuntimeError as e:
    print(f"Simulation error: {e}")
```

### Jetson-Specific Errors

```python
try:
    system.run_inference(inputs)
except MemoryError:
    print("Insufficient memory on Jetson")
except OSError as e:
    print(f"System error: {e}")
```

## Performance Tips

### Memory Optimization

```python
# Use smaller data types
import numpy as np
np.float32  # Instead of np.float64

# Clear unused variables
import gc
gc.collect()

# Monitor memory usage
import psutil
memory = psutil.virtual_memory()
print(f"Memory usage: {memory.percent}%")
```

### Jetson Optimization

```python
# Reduce network size for Jetson
network_config = {
    'max_neurons': 500,  # Reduced from 1000
    'max_synapses': 5000,  # Reduced from 10000
    'connection_probability': 0.05  # Reduced from 0.1
}

# Monitor temperature
if system_info['temperature'] > 70:
    print("Warning: High temperature detected")
```

## Examples

### Complete Network Example

```python
from api.neuromorphic_api import NeuromorphicAPI

# Create network
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
for layer_name, spikes in results['layer_spike_times'].items():
    print(f"{layer_name}: {len(spikes)} spikes")
```

### Jetson Deployment Example

```python
from jetson_optimization import JetsonSensorimotorSystem

# Initialize Jetson system
system = JetsonSensorimotorSystem(use_gpu=True)
system.initialize()

# Run inference
inputs = {
    'vision': np.random.rand(16, 16),
    'auditory': np.random.randn(100),
    'tactile': np.random.rand(8, 8)
}

results = system.run_inference(inputs, duration=50.0)

# Get performance metrics
metrics = results.get('jetson_metrics', {})
print(f"CPU: {metrics.get('current_cpu', 0):.1f}%")
print(f"Memory: {metrics.get('current_memory', 0):.1f}%")
print(f"Temperature: {metrics.get('current_temperature', 0):.1f}°C")
``` 