# Sensory Encoding

Learn how to convert real-world sensory data into spike trains for neuromorphic processing.

## Overview

Sensory encoding is the process of converting continuous sensory signals (images, sounds, touch) into discrete spike trains that spiking neural networks can process. This tutorial covers the various encoding methods available in the system.

## Encoding Types

### 1. Rate Encoding

The simplest encoding method where input intensity is converted to spike frequency.

```python
from core.encoding import RateEncoder

# Create rate encoder
encoder = RateEncoder(num_neurons=100, max_rate=100.0)

# Encode scalar value to spike train
intensity = 0.8  # 80% intensity
spike_times = encoder.encode(intensity, duration=100.0)
print(f"Generated {len(spike_times)} spikes")
```

**Use cases:**
- Simple sensory inputs
- Analog sensor data
- Continuous variables

### 2. Temporal Encoding

Encodes information in precise spike timing rather than rate.

```python
from core.encoding import TemporalEncoder

# Create temporal encoder
encoder = TemporalEncoder(num_neurons=100, time_window=10.0)

# Encode value as spike latency
value = 0.3  # Lower values spike earlier
spike_times = encoder.encode(value)
```

**Use cases:**
- Time-critical information
- High-precision sensory data
- Event-based sensors

### 3. Population Encoding

Distributes information across a population of neurons with different tuning curves.

```python
from core.encoding import PopulationEncoder

# Create population encoder with Gaussian tuning curves
encoder = PopulationEncoder(
    num_neurons=100,
    min_value=0.0,
    max_value=1.0,
    sigma=0.1
)

# Encode value across population
value = 0.5
spike_rates = encoder.encode(value)
```

**Use cases:**
- Continuous variables
- Direction encoding
- Motor commands

## Visual Encoding

### Retinal Encoding

Simulates retinal processing with center-surround receptive fields:

```python
from core.encoding import RetinalEncoder
import numpy as np

# Create retinal encoder
encoder = RetinalEncoder(
    resolution=(32, 32),
    rf_type="on_center",  # or "off_center"
    temporal_resolution=1.0
)

# Encode image
image = np.random.rand(32, 32)  # Grayscale image
spike_trains = encoder.encode(image, duration=100.0)

print(f"Encoded {len(spike_trains)} ganglion cells")
for cell_id, spikes in spike_trains.items():
    print(f"Cell {cell_id}: {len(spikes)} spikes")
```

### Dynamic Vision Sensor (DVS) Encoding

For event-based vision processing:

```python
from core.enhanced_encoding import DVSEncoder

# Create DVS encoder
encoder = DVSEncoder(
    resolution=(128, 128),
    threshold=0.1,  # Intensity change threshold
    refractory_period=1.0
)

# Process video frames
previous_frame = np.zeros((128, 128))
current_frame = np.random.rand(128, 128)

events = encoder.encode_difference(previous_frame, current_frame)
print(f"Generated {len(events)} events")
```

## Auditory Encoding

### Cochlear Encoding

Simulates cochlear processing with frequency decomposition:

```python
from core.enhanced_encoding import CochlearEncoder
import numpy as np

# Create cochlear encoder
encoder = CochlearEncoder(
    num_channels=32,
    freq_range=(20, 20000),  # Hz
    sample_rate=44100
)

# Encode audio signal
duration = 1.0  # seconds
t = np.linspace(0, duration, int(44100 * duration))
audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone

spike_trains = encoder.encode(audio)
print(f"Encoded into {len(spike_trains)} frequency channels")
```

### Onset Detection

For detecting sound onsets and transients:

```python
from core.encoding import OnsetEncoder

# Create onset encoder
encoder = OnsetEncoder(
    threshold=0.1,
    adaptation_rate=0.01
)

# Detect onsets in audio
onsets = encoder.encode(audio)
print(f"Detected {len(onsets)} onset events")
```

## Tactile Encoding

### Mechanoreceptor Encoding

Simulates different types of mechanoreceptors:

```python
from core.enhanced_encoding import MechanoreceptorEncoder

# Create encoder for different receptor types
sa1_encoder = MechanoreceptorEncoder(
    receptor_type="SA1",  # Slowly adapting type 1
    num_receptors=100
)

ra_encoder = MechanoreceptorEncoder(
    receptor_type="RA",   # Rapidly adapting
    num_receptors=100
)

# Encode pressure signal
pressure = np.random.rand(1000) * 10  # kPa
sa1_spikes = sa1_encoder.encode(pressure)
ra_spikes = ra_encoder.encode(pressure)

print(f"SA1: {len(sa1_spikes)} spikes (sustained response)")
print(f"RA: {len(ra_spikes)} spikes (transient response)")
```

## Multimodal Integration

Combining multiple sensory modalities:

```python
from api.neuromorphic_api import NeuromorphicAPI

# Create network with multiple sensory inputs
api = NeuromorphicAPI()
api.create_network()

# Add sensory layers for different modalities
api.add_sensory_layer("visual", 1024, "rate")
api.add_sensory_layer("auditory", 256, "temporal")
api.add_sensory_layer("tactile", 100, "population")

# Add integration layer
api.add_processing_layer("integration", 500, "adex")

# Connect all sensory layers to integration
api.connect_layers("visual", "integration", "feedforward")
api.connect_layers("auditory", "integration", "feedforward")
api.connect_layers("tactile", "integration", "feedforward")

# Prepare multimodal input
visual_input = [(i, t) for i in range(100) for t in np.random.uniform(0, 100, 5)]
auditory_input = [(i, t) for i in range(50) for t in np.random.uniform(0, 100, 10)]
tactile_input = [(i, t) for i in range(20) for t in np.random.uniform(0, 100, 8)]

# Run simulation with multimodal input
results = api.run_simulation(
    duration=100.0,
    external_inputs={
        "visual": visual_input,
        "auditory": auditory_input,
        "tactile": tactile_input
    }
)
```

## Custom Encoders

Creating your own encoder for specific applications:

```python
from core.encoding import Encoder
import numpy as np

class CustomEncoder(Encoder):
    """Custom encoder for specific sensor type."""
    
    def __init__(self, num_neurons, **params):
        super().__init__()
        self.num_neurons = num_neurons
        self.params = params
    
    def encode(self, data, duration=100.0):
        """Convert data to spike trains."""
        spike_trains = {}
        
        for neuron_id in range(self.num_neurons):
            # Custom encoding logic
            rate = self._compute_rate(data, neuron_id)
            num_spikes = np.random.poisson(rate * duration / 1000.0)
            spike_times = np.sort(np.random.uniform(0, duration, num_spikes))
            spike_trains[neuron_id] = spike_times.tolist()
        
        return spike_trains
    
    def _compute_rate(self, data, neuron_id):
        """Compute firing rate for specific neuron."""
        # Implement custom logic
        return float(data * (neuron_id + 1) / self.num_neurons)

# Use custom encoder
encoder = CustomEncoder(num_neurons=50)
spikes = encoder.encode(data=0.7, duration=100.0)
```

## Performance Considerations

### Encoding Resolution
- Higher resolution → more neurons → better representation but higher computational cost
- Balance accuracy with efficiency

### Temporal Precision
- Fine temporal resolution captures fast dynamics
- Coarser resolution reduces computational load

### Sparse Coding
- Use sparse representations when possible
- Reduces energy consumption and improves efficiency

## Practical Examples

### Example 1: Image Classification

```python
# Load and encode image
from PIL import Image
import numpy as np

# Load image
img = Image.open("sample.jpg").convert('L')
img_array = np.array(img.resize((32, 32))) / 255.0

# Encode with retinal encoder
encoder = RetinalEncoder(resolution=(32, 32))
spikes = encoder.encode(img_array, duration=100.0)

# Feed to network
api = NeuromorphicAPI()
api.create_network()
api.add_sensory_layer("retina", 1024, "custom")
api.add_processing_layer("v1", 256, "adex")
api.add_motor_layer("classification", 10)

api.connect_layers("retina", "v1", "feedforward")
api.connect_layers("v1", "classification", "feedforward")

results = api.run_simulation(100.0, external_inputs={"retina": spikes})
```

### Example 2: Sound Recognition

```python
# Process audio for keyword detection
import soundfile as sf

# Load audio
audio, sr = sf.read("keyword.wav")

# Encode with cochlear model
encoder = CochlearEncoder(num_channels=64, sample_rate=sr)
spikes = encoder.encode(audio)

# Process for recognition
api = NeuromorphicAPI()
api.create_network()
api.add_sensory_layer("cochlea", 64, "custom")
api.add_processing_layer("auditory_cortex", 128, "izhikevich")
api.add_motor_layer("keywords", 10)

# ... continue with network setup and simulation
```

## Next Steps

- Learn about [Learning and Plasticity](03_learning_plasticity.md)
- Explore [Neuromodulation](04_neuromodulation.md)
- Implement [Complex Networks](06_complex_networks.md)

---

*← [Getting Started](01_getting_started.md) | [Learning and Plasticity →](03_learning_plasticity.md)*
