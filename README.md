# True Neuromorphic Programming System

A comprehensive proof-of-concept neuromorphic programming system that breaks away from classical artificial neural networks and embraces the temporal, structural, and plastic complexity of biological brains.

## Overview

This system implements a true neuromorphic computing approach with:

- **Biologically Plausible Neuron Models**: Adaptive Exponential Integrate-and-Fire (AdEx), Hodgkin-Huxley, and Leaky Integrate-and-Fire neurons
- **Advanced Synaptic Plasticity**: Spike-Timing-Dependent Plasticity (STDP), short-term plasticity, and neuromodulatory learning
- **Event-Driven Processing**: Asynchronous, temporally precise spike-based computation
- **Neuromodulatory Systems**: Dopamine, serotonin, acetylcholine, and norepinephrine for reward-based learning and behavioral state control
- **Sensory Encoding**: Biologically plausible encoding for visual, auditory, and tactile inputs
- **Homeostatic Regulation**: Network stability through synaptic scaling and firing rate regulation

## Key Features

### 1. Biological Fidelity
- **Temporal Dynamics**: Millisecond-precision spike timing
- **Plasticity Mechanisms**: STDP, short-term plasticity, and neuromodulatory learning
- **Network Topology**: Hierarchical, modular architecture inspired by mammalian neocortex
- **Event-Driven Processing**: Sparse, asynchronous communication

### 2. Learning and Adaptation
- **STDP Learning**: Spike-timing-dependent synaptic plasticity
- **Reward-Based Learning**: Dopaminergic reinforcement learning
- **Homeostatic Regulation**: Synaptic scaling and firing rate homeostasis
- **Behavioral Flexibility**: Serotonergic mood and behavioral state regulation

### 3. Sensory Processing
- **Visual Encoding**: Retinal-like difference-of-Gaussians receptive fields
- **Auditory Encoding**: Cochlear-like tonotopic frequency mapping
- **Tactile Encoding**: Mechanoreceptor-like pressure mapping
- **Multi-Modal Integration**: Combined sensory processing

### 4. High-Level API
- **Easy Network Construction**: Builder pattern for rapid prototyping
- **Flexible Architecture**: Modular layer and connection design
- **Comprehensive Visualization**: Spike rasters, weight evolution, and network activity
- **Training Framework**: Complete sensorimotor learning system

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd neuromorphic-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Network Creation

```python
from api.neuromorphic_api import NeuromorphicAPI

# Create a neuromorphic network
api = NeuromorphicAPI()
api.create_network()

# Add layers
api.add_sensory_layer("input", 100, "rate")
api.add_processing_layer("hidden", 50, "adex")
api.add_motor_layer("output", 10)

# Connect layers with STDP learning
api.connect_layers("input", "hidden", "feedforward", synapse_type="stdp")
api.connect_layers("hidden", "output", "feedforward", synapse_type="stdp")

# Run simulation
results = api.run_simulation(100.0, external_inputs={"input": [(i, i*10.0) for i in range(20)]})
```

### Sensorimotor System

```python
from api.neuromorphic_api import SensorimotorSystem

# Create complete sensorimotor system
system = SensorimotorSystem()

# Train the system
training_data = create_training_data(num_trials=50)
training_results = system.train(training_data, epochs=100)

# Run a trial
sensory_inputs = {
    'visual': create_visual_input(),
    'auditory': create_auditory_input(),
    'tactile': create_tactile_input()
}
results = system.run_trial(sensory_inputs)
```

### Sensory Encoding

```python
from core.encoding import RetinalEncoder, CochlearEncoder, SomatosensoryEncoder

# Visual encoding
visual_encoder = RetinalEncoder()
image = create_visual_input()
visual_spikes = visual_encoder.encode(image)

# Auditory encoding
auditory_encoder = CochlearEncoder()
audio = create_auditory_input()
auditory_spikes = auditory_encoder.encode(audio)

# Tactile encoding
tactile_encoder = SomatosensoryEncoder()
pressure_map = create_tactile_input()
tactile_spikes = tactile_encoder.encode(pressure_map)
```

## System Architecture

### Core Components

1. **Neuron Models** (`core/neurons.py`)
   - `AdaptiveExponentialIntegrateAndFire`: Biologically realistic neuron model
   - `HodgkinHuxleyNeuron`: Full ion channel dynamics
   - `LeakyIntegrateAndFire`: Computationally efficient model

2. **Synapse Models** (`core/synapses.py`)
   - `STDP_Synapse`: Spike-timing-dependent plasticity
   - `ShortTermPlasticitySynapse`: Depression and facilitation
   - `NeuromodulatorySynapse`: Reward-modulated plasticity

3. **Network Architecture** (`core/network.py`)
   - `NeuromorphicNetwork`: Complete network with layers and connections
   - `EventDrivenSimulator`: Event-driven simulation engine
   - `NetworkBuilder`: Helper for network construction

4. **Sensory Encoding** (`core/encoding.py`)
   - `RetinalEncoder`: Visual encoding with DoG receptive fields
   - `CochlearEncoder`: Auditory encoding with tonotopic mapping
   - `SomatosensoryEncoder`: Tactile encoding with pressure mapping

5. **Neuromodulation** (`core/neuromodulation.py`)
   - `DopaminergicSystem`: Reward prediction error and reinforcement learning
   - `SerotonergicSystem`: Mood and behavioral state regulation
   - `CholinergicSystem`: Attention and learning rate modulation
   - `NoradrenergicSystem`: Arousal and vigilance control

6. **High-Level API** (`api/neuromorphic_api.py`)
   - `NeuromorphicAPI`: Easy-to-use interface for network construction
   - `SensorimotorSystem`: Complete sensorimotor control system
   - `NeuromorphicVisualizer`: Comprehensive visualization tools

## Demonstration

Run the demonstration script to see the system in action:

```bash
python demo/sensorimotor_demo.py
```

This will show:
- Basic network functionality
- Sensory encoding capabilities
- Adaptive behavior with STDP learning
- Complete sensorimotor learning system

## Biological Foundations

### Neuron Dynamics
The system implements the Adaptive Exponential Integrate-and-Fire (AdEx) model:

```
τ_m * dV/dt = -(V - E_L) + ΔT * exp((V - V_T)/ΔT) - w + I_syn
τ_w * dw/dt = a * (V - E_L) - w
```

### Synaptic Plasticity
STDP learning rule with timing-dependent weight updates:

```
Δw = A_+ * exp(-Δt/τ_+) for Δt > 0 (LTP)
Δw = -A_- * exp(Δt/τ_-) for Δt < 0 (LTD)
```

### Neuromodulatory Systems
- **Dopamine**: Reward prediction error and reinforcement learning
- **Serotonin**: Mood regulation and behavioral flexibility
- **Acetylcholine**: Attention and learning rate modulation
- **Norepinephrine**: Arousal and vigilance control

## Hardware Compatibility

The system is designed to be compatible with emerging neuromorphic hardware:

- **Intel Loihi**: Event-driven neuron cores and synaptic plasticity engines
- **IBM TrueNorth**: Crossbar architecture with configurable neuron models
- **BrainScaleS**: Mixed-signal neuromorphic hardware

## Evaluation Metrics

The system provides comprehensive evaluation metrics:

- **Biological Fidelity**: Spike train similarity and learning convergence
- **Performance**: Energy efficiency, latency, and throughput
- **Learning**: Reward convergence and behavioral adaptation
- **Robustness**: Noise tolerance and fault resilience

## Future Directions

### Planned Enhancements
1. **Glial Cell Integration**: Astrocyte-neuron interactions
2. **Structural Plasticity**: Dynamic synapse formation/elimination
3. **Meta-Plasticity**: Higher-order learning rules
4. **Cognitive Integration**: Working memory and attention mechanisms

### Research Applications
- **Neuroscience Research**: Biological neural network modeling
- **Robotics**: Adaptive sensorimotor control
- **Brain-Computer Interfaces**: Neural decoding and encoding
- **Neuromorphic Computing**: Energy-efficient AI systems

## Contributing

We welcome contributions to advance the state of neuromorphic computing:

1. **Biological Modeling**: Enhanced neuron and synapse models
2. **Hardware Integration**: Platform-specific optimizations
3. **Applications**: Novel use cases and demonstrations
4. **Documentation**: Improved tutorials and examples

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

1. Brette, R., & Gerstner, W. (2005). Adaptive exponential integrate-and-fire model as an effective description of neuronal activity. Journal of neurophysiology, 94(5), 3637-3642.

2. Bi, G. Q., & Poo, M. M. (1998). Synaptic modifications in cultured hippocampal neurons: dependence on spike timing, synaptic strength, and postsynaptic cell type. Journal of neuroscience, 18(24), 10464-10472.

3. Markram, H., et al. (2015). Reconstruction and simulation of neocortical microcircuitry. Cell, 163(2), 456-492.

4. Davies, M., et al. (2018). Loihi: A neuromorphic manycore processor with on-chip learning. IEEE Micro, 38(1), 82-99.

## Contact

For questions, suggestions, or collaborations, please open an issue or contact the development team.

---

*This system represents a significant step toward true neuromorphic computing, bridging the gap between biological neuroscience and practical computing applications.* 