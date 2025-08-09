# Demo Guide: Enhanced Neuromorphic System

## Overview

This guide covers all demonstration systems in the Enhanced Neuromorphic System, from basic functionality to advanced hardware-specific implementations.

## 🎯 Demo Categories

### 1. Enhanced Comprehensive Demo
**File:** `demo/enhanced_comprehensive_demo.py`
**Purpose:** Complete system showcase with all enhanced features

#### Features Demonstrated
- ✅ **Progressive Task Complexity** (7 levels)
- ✅ **Enhanced Sensory Encoding** (multi-modal)
- ✅ **Comprehensive Robustness Testing**
- ✅ **Dynamic Neural Logging**
- ✅ **Automated Analysis & Visualization**

#### Usage
```python
from demo.enhanced_comprehensive_demo import EnhancedNeuromorphicDemo

# Initialize demo
demo = EnhancedNeuromorphicDemo()

# Run complete demonstration
demo.run_full_demo()

# Or run individual components
demo.run_progressive_task_complexity_demo()
demo.run_enhanced_sensory_encoding_demo()
demo.run_comprehensive_robustness_testing()
demo.run_dynamic_logging_demo()
demo.run_comprehensive_analysis()
```

#### Outputs Generated
- **Neural Data**: `neural_data/` directory with JSON exports
- **Visualizations**: `plots/` directory with analysis plots
- **Reports**: `reports/` directory with performance summaries
- **Logs**: `enhanced_trace.log` with detailed system activity

### 2. Hardware-Specific Demos

#### GPU Large Scale Demo
**File:** `demo/gpu_large_scale_demo.py`
**Purpose:** GPU-optimized large-scale simulations

#### Features
- 🚀 **CUDA/OpenCL Acceleration**
- 📊 **Large-scale Network Simulations**
- ⚡ **High-performance Computing**
- 🔧 **GPU Memory Optimization**

#### Usage
```python
from demo.gpu_large_scale_demo import run_gpu_demo

# Run GPU-optimized demo
run_gpu_demo()

# Or with custom parameters
run_gpu_demo(
    network_size=10000,
    simulation_time=1000.0,
    gpu_memory_limit="8GB"
)
```

#### Requirements
- NVIDIA GPU with CUDA support
- `requirements_gpu.txt` dependencies
- Sufficient GPU memory (4GB+ recommended)

#### Jetson Demo
**File:** `demo/jetson_demo.py`
**Purpose:** NVIDIA Jetson edge computing optimization

#### Features
- 🔧 **Jetson-specific Optimizations**
- 📱 **Edge Computing Capabilities**
- ⚡ **Real-time Processing**
- 🔋 **Power Efficiency**

#### Usage
```python
from demo.jetson_demo import run_jetson_demo

# Run Jetson-optimized demo
run_jetson_demo()

# Or with custom parameters
run_jetson_demo(
    power_mode="MAXN",
    gpu_freq=921600,
    cpu_freq=1479000
)
```

#### Requirements
- NVIDIA Jetson device (Nano, Xavier, Orin)
- `requirements_jetson.txt` dependencies
- JetPack SDK installed

#### Sensorimotor Demo
**File:** `demo/sensorimotor_demo.py`
**Purpose:** Real-world sensor integration and motor control

#### Features
- 🎯 **Sensor Integration** (camera, microphone, touch)
- 🤖 **Motor Control Simulation**
- 🔄 **Real-world Interaction**
- 📊 **Sensor Data Processing**

#### Usage
```python
from demo.sensorimotor_demo import run_sensorimotor_demo

# Run sensorimotor demo
run_sensorimotor_demo()

# Or with specific sensors
run_sensorimotor_demo(
    enable_camera=True,
    enable_microphone=True,
    enable_touch=True
)
```

#### Requirements
- Camera (USB or built-in)
- Microphone
- Touch sensor (optional)
- Motor control interface (simulated)

#### Sensorimotor Training
**File:** `demo/sensorimotor_training.py`
**Purpose:** Training sensorimotor coordination

#### Features
- 🎓 **Learning Sensorimotor Coordination**
- 🔄 **Adaptive Behavior**
- 📈 **Performance Tracking**
- 🎯 **Task-specific Training**

#### Usage
```python
from demo.sensorimotor_training import run_sensorimotor_training

# Run training session
run_sensorimotor_training()

# Or with custom training parameters
run_sensorimotor_training(
    training_epochs=100,
    task_complexity="intermediate",
    learning_rate=0.01
)
```

## 🎮 Demo Scenarios

### Scenario 1: Basic System Exploration
**Best for:** New users, understanding core concepts
```bash
# Run the comprehensive demo
python demo/enhanced_comprehensive_demo.py
```

**What you'll see:**
- System initialization and setup
- Progressive task complexity demonstration
- Sensory encoding examples
- Robustness testing results
- Generated visualizations and reports

### Scenario 2: Hardware Performance Testing
**Best for:** Hardware evaluation, performance benchmarking
```bash
# Test GPU performance
python demo/gpu_large_scale_demo.py

# Test Jetson performance
python demo/jetson_demo.py
```

**What you'll see:**
- Hardware detection and optimization
- Performance metrics and benchmarks
- Memory usage and efficiency
- Real-time processing capabilities

### Scenario 3: Real-world Applications
**Best for:** Practical applications, robotics integration
```bash
# Run sensorimotor demo
python demo/sensorimotor_demo.py

# Run training session
python demo/sensorimotor_training.py
```

**What you'll see:**
- Sensor data acquisition
- Real-time processing
- Motor control simulation
- Learning and adaptation

## 📊 Demo Outputs

### Generated Files
```
project_root/
├── neural_data/           # Neural activity data
│   ├── spike_events.json
│   ├── membrane_potentials.json
│   └── synaptic_weights.json
├── plots/                 # Visualization outputs
│   ├── spike_raster.png
│   ├── membrane_potentials.png
│   ├── firing_rates.png
│   └── network_state.png
├── reports/               # Performance reports
│   ├── robustness_summary.json
│   ├── performance_metrics.json
│   └── system_health.json
└── enhanced_trace.log     # Detailed system logs
```

### Analysis Plots
1. **Spike Raster Plot**: Neural firing patterns over time
2. **Membrane Potential Traces**: Neuron voltage dynamics
3. **Firing Rate Histograms**: Activity distribution analysis
4. **Synaptic Weight Evolution**: Learning progress visualization
5. **Network State Overview**: System health monitoring

### Performance Reports
1. **Robustness Summary**: System stress test results
2. **Performance Metrics**: Speed, accuracy, efficiency
3. **System Health**: Overall system status and recommendations

## 🔧 Customization

### Modifying Demo Parameters
```python
# Customize comprehensive demo
demo = EnhancedNeuromorphicDemo()

# Modify network parameters
demo.network_params = {
    'input_size': 500,
    'hidden_size': 200,
    'output_size': 50
}

# Modify task complexity
demo.task_levels = [TaskLevel.LEVEL_1, TaskLevel.LEVEL_3, TaskLevel.LEVEL_5]

# Run with custom parameters
demo.run_full_demo()
```

### Adding Custom Sensors
```python
# Extend sensorimotor demo
from demo.sensorimotor_demo import SensorimotorDemo

demo = SensorimotorDemo()

# Add custom sensor
demo.add_sensor('custom_sensor', CustomSensorClass())

# Add custom motor
demo.add_motor('custom_motor', CustomMotorClass())

# Run with custom components
demo.run_demo()
```

### Hardware Integration
```python
# GPU demo with custom settings
from demo.gpu_large_scale_demo import GPUDemo

demo = GPUDemo()
demo.set_gpu_settings(
    memory_limit="16GB",
    compute_capability="8.6",
    optimization_level="MAX"
)
demo.run_demo()
```

## 🚀 Advanced Usage

### Batch Processing
```python
# Run multiple demos in sequence
from demo.enhanced_comprehensive_demo import EnhancedNeuromorphicDemo
from demo.gpu_large_scale_demo import run_gpu_demo
from demo.jetson_demo import run_jetson_demo

# Run comprehensive demo
comprehensive_demo = EnhancedNeuromorphicDemo()
comprehensive_demo.run_full_demo()

# Run hardware-specific demos
run_gpu_demo()
run_jetson_demo()
```

### Performance Comparison
```python
# Compare different configurations
configurations = [
    {'network_size': 1000, 'task_level': TaskLevel.LEVEL_1},
    {'network_size': 5000, 'task_level': TaskLevel.LEVEL_3},
    {'network_size': 10000, 'task_level': TaskLevel.LEVEL_5}
]

for config in configurations:
    demo = EnhancedNeuromorphicDemo()
    demo.network_params = config
    results = demo.run_full_demo()
    print(f"Config {config}: {results['performance_score']}")
```

### Research Integration
```python
# Use demos for research experiments
from demo.enhanced_comprehensive_demo import EnhancedNeuromorphicDemo

# Setup experiment
demo = EnhancedNeuromorphicDemo()
demo.setup_experiment(
    experiment_name="robustness_study",
    parameters={
        'noise_levels': [0.0, 0.1, 0.2, 0.3],
        'missing_modalities': ['visual', 'auditory', 'tactile'],
        'adversarial_attacks': ['FGSM', 'PGD', 'universal']
    }
)

# Run experiment
results = demo.run_experiment()

# Export results
demo.export_results('research_output.json')
```

## 🔍 Troubleshooting

### Common Demo Issues

#### Demo Won't Start
```bash
# Check dependencies
pip install -r requirements.txt

# Check Python version
python --version  # Should be 3.8+

# Check imports
python -c "from demo.enhanced_comprehensive_demo import EnhancedNeuromorphicDemo"
```

#### GPU Demo Issues
```bash
# Check CUDA installation
nvidia-smi

# Check PyTorch CUDA support
python -c "import torch; print(torch.cuda.is_available())"

# Install GPU requirements
pip install -r requirements_gpu.txt
```

#### Jetson Demo Issues
```bash
# Check Jetson device
cat /etc/nv_tegra_release

# Check JetPack version
dpkg -l | grep nvidia-l4t

# Install Jetson requirements
pip install -r requirements_jetson.txt
```

#### Performance Issues
```python
# Reduce demo complexity
demo = EnhancedNeuromorphicDemo()
demo.network_params = {'input_size': 100, 'hidden_size': 50}
demo.task_levels = [TaskLevel.LEVEL_1]
demo.run_full_demo()
```

## 📚 Next Steps

### For Beginners
1. Start with `enhanced_comprehensive_demo.py`
2. Review generated visualizations and reports
3. Experiment with different parameters
4. Read the core system documentation

### For Advanced Users
1. Explore hardware-specific demos
2. Customize demo parameters for your use case
3. Integrate with your own sensors and hardware
4. Contribute improvements to the demo system

### For Researchers
1. Use demos as baseline for experiments
2. Extend demos with custom research components
3. Publish findings using the demo framework
4. Collaborate on demo improvements

---

**Ready to explore?** Start with `python demo/enhanced_comprehensive_demo.py` to see the full system in action! 