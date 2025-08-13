# Edge Deployment

Deploy neuromorphic networks on edge devices including NVIDIA Jetson Nano and other embedded systems.

## Overview

Edge deployment enables neuromorphic systems to run efficiently on resource-constrained devices, bringing brain-inspired computing to IoT, robotics, and embedded applications.

## Jetson Nano Deployment

### System Requirements

- NVIDIA Jetson Nano (2GB or 4GB model)
- JetPack SDK 4.6 or later
- Python 3.8+
- CUDA 10.2 (included with JetPack)
- cuDNN 8.0 (included with JetPack)

### Installation on Jetson

```bash
# Update system
sudo apt-get update
sudo apt-get upgrade

# Install Python dependencies
pip3 install numpy scipy matplotlib

# Install CUDA-aware packages
pip3 install cupy-cuda102

# Clone and install neuromorphic system
git clone <repository-url>
cd neuron
pip3 install -e ".[jetson]"
```

### Performance Optimization

```python
from scripts.jetson_optimization import JetsonOptimizer

# Initialize optimizer
optimizer = JetsonOptimizer()

# Check system capabilities
info = optimizer.get_system_info()
print(f"Available memory: {info['memory_available']} MB")
print(f"GPU cores: {info['gpu_cores']}")
print(f"Max frequency: {info['max_frequency']} MHz")

# Set power mode
optimizer.set_power_mode("MAXN")  # Maximum performance
# or "5W" for low power mode

# Enable GPU acceleration
optimizer.enable_gpu_acceleration()
```

### Memory-Efficient Networks

```python
from api.neuromorphic_api import NeuromorphicAPI

# Create memory-optimized network
api = NeuromorphicAPI(device="jetson")

# Use smaller precision
api.set_precision("float16")  # Half precision

# Create compact network
api.create_network(max_neurons=500, max_synapses=5000)

# Use efficient neuron models
api.add_sensory_layer("input", 100, "lif")  # LIF is memory-efficient
api.add_processing_layer("hidden", 50, "lif")
api.add_motor_layer("output", 10)

# Sparse connectivity
api.connect_layers("input", "hidden", "feedforward", 
                  connection_probability=0.1)  # Only 10% connections

# Run with minimal recording
results = api.run_simulation(
    duration=100.0,
    record_spikes=True,
    record_potentials=False,  # Save memory
    record_weights=False
)
```

## Optimization Strategies

### 1. Network Pruning

Remove unnecessary connections:

```python
from core.optimization import NetworkPruner

# Create pruner
pruner = NetworkPruner(
    threshold=0.01,  # Remove weights below threshold
    sparsity_target=0.8  # Target 80% sparsity
)

# Prune network
pruned_network = pruner.prune(network)
print(f"Connections reduced: {pruner.reduction_ratio:.1%}")

# Fine-tune after pruning
api.fine_tune(pruned_network, epochs=10)
```

### 2. Quantization

Reduce numerical precision:

```python
from core.optimization import Quantizer

# Create quantizer
quantizer = Quantizer()

# Quantize weights to 8-bit integers
quantized_weights = quantizer.quantize_weights(
    network.get_weights(),
    bits=8,
    symmetric=True
)

# Quantize neuron states
quantizer.quantize_neurons(
    network,
    voltage_bits=16,
    current_bits=8
)

# Measure accuracy impact
original_accuracy = evaluate(network)
quantized_accuracy = evaluate(quantized_network)
print(f"Accuracy change: {quantized_accuracy - original_accuracy:.2%}")
```

### 3. Event-Driven Processing

Minimize computation with event-based simulation:

```python
from engine import Network, Simulator, SimulationMode

# Create event-driven network
network = Network("EdgeNetwork")
network.add_neuron_group("input", 100, "lif")
network.add_neuron_group("output", 10, "lif")
network.connect("input", "output", connectivity=0.1)

# Use event-driven simulator
sim = Simulator(network, mode=SimulationMode.EVENT_DRIVEN)

# Process only when events occur
sim.set_event_threshold(0.01)  # Ignore small changes

# Run efficiently
results = sim.run(duration=1000.0, min_timestep=0.1)
```

## Real-Time Processing

### Camera Input

Process video in real-time:

```python
import cv2
from core.encoding import DVSEncoder

# Initialize camera
cap = cv2.VideoCapture(0)  # USB camera
# or for CSI camera on Jetson:
# cap = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=480, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")

# Create DVS encoder
encoder = DVSEncoder(resolution=(64, 64), threshold=0.1)

# Create processing network
api = NeuromorphicAPI(device="jetson")
api.create_network()
api.add_sensory_layer("retina", 64*64, "lif")
api.add_processing_layer("v1", 256, "lif")
api.add_motor_layer("detection", 10)

# Real-time processing loop
prev_frame = None
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to grayscale and resize
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    
    # Generate events
    if prev_frame is not None:
        events = encoder.encode_difference(prev_frame, resized)
        
        # Process with network
        if len(events) > 0:
            results = api.run_simulation(
                duration=10.0,
                external_inputs={"retina": events}
            )
            
            # Get detection output
            detection = results['layer_spike_counts']['detection']
            print(f"Detected class: {np.argmax(detection)}")
    
    prev_frame = resized
    
    # Display (optional)
    cv2.imshow('Input', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Sensor Integration

Interface with sensors:

```python
import board
import busio
import adafruit_mpu6050

# Initialize I2C sensors
i2c = busio.I2C(board.SCL, board.SDA)
mpu = adafruit_mpu6050.MPU6050(i2c)

# Create sensory processing network
api = NeuromorphicAPI(device="jetson")
api.create_network()
api.add_sensory_layer("imu", 6, "rate")
api.add_processing_layer("integration", 20, "adex")
api.add_motor_layer("balance", 4)

# Sensor processing loop
while True:
    # Read sensor data
    accel = mpu.acceleration
    gyro = mpu.gyro
    
    # Encode sensor values
    sensor_data = list(accel) + list(gyro)
    spikes = encode_sensor_data(sensor_data)
    
    # Process with network
    results = api.run_simulation(
        duration=10.0,
        external_inputs={"imu": spikes}
    )
    
    # Get motor commands
    motor_output = results['layer_rates']['balance']
    send_motor_commands(motor_output)
    
    time.sleep(0.01)  # 100 Hz update rate
```

## Power Management

### Dynamic Power Scaling

Adjust performance based on battery:

```python
from scripts.jetson_optimization import PowerManager

# Initialize power manager
pm = PowerManager()

# Monitor battery level
battery_level = pm.get_battery_level()

if battery_level < 20:
    # Low power mode
    pm.set_power_mode("5W")
    api.set_simulation_quality("low")
elif battery_level < 50:
    # Balanced mode
    pm.set_power_mode("10W")
    api.set_simulation_quality("medium")
else:
    # Full performance
    pm.set_power_mode("MAXN")
    api.set_simulation_quality("high")

# Thermal management
temp = pm.get_temperature()
if temp > 70:  # Celsius
    pm.enable_thermal_throttling()
```

### Sleep Modes

Implement power-saving sleep cycles:

```python
# Configure sleep behavior
api.set_sleep_mode(
    enabled=True,
    idle_threshold=1.0,  # Sleep after 1 second idle
    wake_on_input=True
)

# Processing with automatic sleep
while True:
    if api.has_input():
        # Wake and process
        api.wake()
        results = api.process_input()
        
        # Return to sleep if idle
        if api.is_idle():
            api.sleep()
    
    time.sleep(0.001)  # Minimal polling
```

## Deployment Examples

### Robot Control

Deploy on mobile robot:

```python
# Robot control network
api = NeuromorphicAPI(device="jetson")
api.create_network()

# Sensory inputs
api.add_sensory_layer("vision", 1024, "lif")
api.add_sensory_layer("lidar", 360, "rate")
api.add_sensory_layer("imu", 9, "rate")

# Processing layers
api.add_processing_layer("perception", 256, "adex")
api.add_processing_layer("planning", 128, "adex")

# Motor outputs
api.add_motor_layer("wheels", 2)  # Left, right
api.add_motor_layer("arm", 6)     # 6-DOF arm

# Connect with appropriate patterns
api.connect_layers("vision", "perception", "feedforward")
api.connect_layers("lidar", "perception", "feedforward")
api.connect_layers("imu", "planning", "feedforward")
api.connect_layers("perception", "planning", "feedforward")
api.connect_layers("planning", "wheels", "feedforward")
api.connect_layers("planning", "arm", "feedforward")

# Main control loop
while robot.is_active():
    # Get sensor data
    vision_data = robot.get_camera_frame()
    lidar_data = robot.get_lidar_scan()
    imu_data = robot.get_imu_reading()
    
    # Encode and process
    inputs = {
        "vision": encode_vision(vision_data),
        "lidar": encode_lidar(lidar_data),
        "imu": encode_imu(imu_data)
    }
    
    results = api.run_simulation(20.0, external_inputs=inputs)
    
    # Execute motor commands
    wheel_commands = decode_motor(results['layer_rates']['wheels'])
    arm_commands = decode_motor(results['layer_rates']['arm'])
    
    robot.set_wheel_velocities(*wheel_commands)
    robot.set_arm_positions(arm_commands)
```

### IoT Sensor Node

Deploy as smart sensor:

```python
# IoT sensor network
api = NeuromorphicAPI(device="jetson")
api.create_network()

# Multiple sensor inputs
api.add_sensory_layer("temperature", 10, "rate")
api.add_sensory_layer("humidity", 10, "rate")
api.add_sensory_layer("motion", 100, "temporal")

# Anomaly detection
api.add_processing_layer("feature", 50, "lif")
api.add_processing_layer("anomaly", 20, "adex")
api.add_motor_layer("alert", 3)  # Normal, warning, critical

# Train on normal patterns
train_anomaly_detector(api, normal_data)

# Deployment
while True:
    # Read sensors
    temp = read_temperature_array()
    humidity = read_humidity_array()
    motion = read_motion_sensor()
    
    # Process
    inputs = {
        "temperature": encode_temperature(temp),
        "humidity": encode_humidity(humidity),
        "motion": encode_motion(motion)
    }
    
    results = api.run_simulation(50.0, external_inputs=inputs)
    
    # Check for anomalies
    alert_level = np.argmax(results['layer_rates']['alert'])
    
    if alert_level > 0:
        send_alert(alert_level)
    
    # Power-efficient sleep
    time.sleep(1.0)
```

## Performance Benchmarks

### Jetson Nano Performance

| Metric | Desktop | Jetson Nano | Optimization |
|--------|---------|-------------|--------------|
| Neurons | 1000+ | 500-1000 | 50% reduction |
| Synapses | 10000+ | 5000-10000 | 50% reduction |
| Simulation Speed | 1000x real-time | 100x real-time | 10x slower |
| Power Usage | 50-100W | 5-10W | 90% reduction |
| Memory | 4-8GB | 2-4GB | 50% reduction |

### Optimization Impact

```python
# Benchmark optimizations
from benchmarks.edge_benchmarks import benchmark_edge

# Test different configurations
configs = [
    {"precision": "float32", "pruning": 0.0},
    {"precision": "float16", "pruning": 0.0},
    {"precision": "float16", "pruning": 0.5},
    {"precision": "int8", "pruning": 0.8}
]

for config in configs:
    results = benchmark_edge(config)
    print(f"Config: {config}")
    print(f"  Speed: {results['speed']} Hz")
    print(f"  Power: {results['power']} W")
    print(f"  Accuracy: {results['accuracy']:.2%}")
```

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce network size
   - Disable recording
   - Use lower precision

2. **Slow Performance**
   - Enable GPU acceleration
   - Use event-driven mode
   - Implement pruning

3. **High Power Usage**
   - Use appropriate power mode
   - Implement sleep cycles
   - Reduce update frequency

4. **Thermal Throttling**
   - Add cooling (fan/heatsink)
   - Reduce clock speeds
   - Implement thermal management

## Next Steps

- Explore [Complex Networks](06_complex_networks.md)
- Learn about [Performance Tuning](07_performance_tuning.md)
- Read [Deployment Best Practices](08_deployment_best_practices.md)

---

*← [Neuromodulation](04_neuromodulation.md) | [Complex Networks →](06_complex_networks.md)*
