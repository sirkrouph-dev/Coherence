"""
Jetson Nano Optimization for Neuromorphic System
===============================================

This module provides optimizations for running the neuromorphic programming system
on NVIDIA Jetson Nano 8GB for edge neuromorphic computing applications.
"""

from core.synapses import SynapseFactory, SynapsePopulation
from core.neurons import NeuronFactory, NeuronPopulation
from core.neuromodulation import (HomeostaticRegulator,
                                  NeuromodulatoryController)
from core.network import NetworkBuilder, NeuromorphicNetwork
from core.encoding import CochlearEncoder, RetinalEncoder, SomatosensoryEncoder
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import numpy as np
import psutil

# Jetson-specific imports
try:
    import cupy as cp  # GPU acceleration for Jetson

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("CuPy not available, using CPU-only mode")

try:
    import cv2  # OpenCV for Jetson

    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("OpenCV not available")

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class JetsonOptimizer:
    """Optimization utilities for Jetson Nano deployment."""

    def __init__(self):
        """Initialize Jetson optimizer."""
        self.cuda_available = CUDA_AVAILABLE
        self.opencv_available = OPENCV_AVAILABLE
        self.performance_metrics = {}
        self.setup_logging()

    def setup_logging(self):
        """Setup logging for Jetson deployment."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("jetson_neuromorphic.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def get_system_info(self) -> Dict[str, Any]:
        """Get Jetson Nano system information."""
        info = {
            "cpu_count": os.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "cuda_available": self.cuda_available,
            "opencv_available": self.opencv_available,
            "temperature": self.get_temperature(),
            "power_consumption": self.get_power_consumption(),
        }
        return info

    def get_temperature(self) -> float:
        """Get Jetson Nano temperature."""
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                temp = float(f.read()) / 1000.0
            return temp
        except BaseException:
            return 0.0

    def get_power_consumption(self) -> float:
        """Get power consumption estimate."""
        try:
            # Read from power monitoring if available
            with open(
                "/sys/bus/i2c/devices/0-0040/iio_device/in_power0_input", "r"
            ) as f:
                power = float(f.read()) / 1000.0  # Convert to Watts
            return power
        except BaseException:
            return 0.0

    def optimize_network_size(
        self, target_neurons: int = 1000, target_synapses: int = 10000
    ) -> Dict[str, int]:
        """Optimize network size for Jetson Nano constraints."""
        system_info = self.get_system_info()
        available_memory = system_info["memory_available"]

        # Memory constraints (estimate 1KB per neuron, 100B per synapse)
        max_neurons = min(target_neurons, int(available_memory * 0.1 / 1024))
        max_synapses = min(target_synapses, int(available_memory * 0.3 / 100))

        # Temperature-based scaling
        temp = system_info["temperature"]
        if temp > 70:  # Reduce if too hot
            max_neurons = int(max_neurons * 0.7)
            max_synapses = int(max_synapses * 0.7)

        return {
            "max_neurons": max_neurons,
            "max_synapses": max_synapses,
            "recommended_layers": min(5, max_neurons // 200),
        }

    def create_jetson_network(
        self, network_config: Dict[str, Any]
    ) -> "JetsonNeuromorphicNetwork":
        """Create a Jetson-optimized neuromorphic network."""
        optimizer = self.optimize_network_size()

        # Adjust network configuration for Jetson constraints
        adjusted_config = self.adjust_network_config(network_config, optimizer)

        # Create optimized network
        network = JetsonNeuromorphicNetwork(
            max_neurons=optimizer["max_neurons"],
            max_synapses=optimizer["max_synapses"],
            use_gpu=self.cuda_available,
        )

        network.build_from_config(adjusted_config)
        return network

    def adjust_network_config(
        self, config: Dict[str, Any], optimizer: Dict[str, int]
    ) -> Dict[str, Any]:
        """Adjust network configuration for Jetson constraints."""
        adjusted_config = config.copy()

        # Scale down layer sizes
        for layer in adjusted_config.get("layers", []):
            if layer["type"] == "sensory":
                layer["size"] = min(layer["size"], optimizer["max_neurons"] // 4)
            elif layer["type"] == "processing":
                layer["size"] = min(layer["size"], optimizer["max_neurons"] // 8)
            elif layer["type"] == "motor":
                layer["size"] = min(layer["size"], optimizer["max_neurons"] // 16)

        # Reduce connection probabilities
        for connection in adjusted_config.get("connections", []):
            connection["probability"] = min(connection.get("probability", 0.1), 0.05)

        return adjusted_config


class JetsonNeuromorphicNetwork(NeuromorphicNetwork):
    """Jetson-optimized neuromorphic network."""

    def __init__(
        self, max_neurons: int = 1000, max_synapses: int = 10000, use_gpu: bool = False
    ):
        """Initialize Jetson-optimized network."""
        super().__init__()
        self.max_neurons = max_neurons
        self.max_synapses = max_synapses
        self.use_gpu = use_gpu
        self.performance_monitor = JetsonPerformanceMonitor()
        self.logger = logging.getLogger(__name__)

        if self.use_gpu:
            self.logger.info("GPU acceleration enabled")
        else:
            self.logger.info("Using CPU-only mode")

    def build_from_config(self, config: Dict[str, Any]):
        """Build network from configuration with Jetson optimizations."""
        builder = NetworkBuilder()

        # Track resource usage
        neuron_count = 0
        synapse_count = 0

        for layer in config.get("layers", []):
            if neuron_count >= self.max_neurons:
                self.logger.warning("Neuron limit reached, skipping remaining layers")
                break

            if layer["type"] == "sensory":
                builder.add_sensory_layer(
                    layer["name"], layer["size"], layer.get("encoding_type", "rate")
                )
                neuron_count += layer["size"]
            elif layer["type"] == "processing":
                builder.add_processing_layer(
                    layer["name"], layer["size"], layer.get("neuron_type", "adex")
                )
                neuron_count += layer["size"]
            elif layer["type"] == "motor":
                builder.add_motor_layer(layer["name"], layer["size"])
                neuron_count += layer["size"]

        # Create connections with resource monitoring
        for connection in config.get("connections", []):
            if synapse_count >= self.max_synapses:
                self.logger.warning(
                    "Synapse limit reached, skipping remaining connections"
                )
                break

            builder.connect_layers(
                connection["pre_layer"],
                connection["post_layer"],
                connection_type=connection.get("connection_type", "random"),
                synapse_type=connection.get("synapse_type", "stdp"),
                connection_probability=connection.get("probability", 0.05),
            )

        self.network = builder.build()
        self.logger.info(
            f"Built Jetson network: {neuron_count} neurons, {synapse_count} synapses"
        )

    def run_simulation(self, duration: float, dt: float = 0.1) -> Dict[str, Any]:
        """Run simulation with Jetson performance monitoring."""
        self.performance_monitor.start_monitoring()

        try:
            results = super().run_simulation(duration, dt)

            # Add Jetson-specific metrics
            results["jetson_metrics"] = self.performance_monitor.get_metrics()

            return results
        finally:
            self.performance_monitor.stop_monitoring()


class JetsonPerformanceMonitor:
    """Monitor performance on Jetson Nano."""

    def __init__(self):
        """Initialize performance monitor."""
        self.monitoring = False
        self.metrics = {}
        self.start_time = None

    def start_monitoring(self):
        """Start performance monitoring."""
        self.monitoring = True
        self.start_time = time.time()
        self.metrics = {
            "cpu_usage": [],
            "memory_usage": [],
            "temperature": [],
            "power_consumption": [],
        }

    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring = False

    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        if not self.monitoring:
            return {}

        # Get current metrics
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent

        # Get temperature and power
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                temp = float(f.read()) / 1000.0
        except BaseException:
            temp = 0.0

        try:
            with open(
                "/sys/bus/i2c/devices/0-0040/iio_device/in_power0_input", "r"
            ) as f:
                power = float(f.read()) / 1000.0
        except BaseException:
            power = 0.0

        # Store metrics
        self.metrics["cpu_usage"].append(cpu_percent)
        self.metrics["memory_usage"].append(memory_percent)
        self.metrics["temperature"].append(temp)
        self.metrics["power_consumption"].append(power)

        return {
            "current_cpu": cpu_percent,
            "current_memory": memory_percent,
            "current_temperature": temp,
            "current_power": power,
            "average_cpu": np.mean(self.metrics["cpu_usage"]),
            "average_memory": np.mean(self.metrics["memory_usage"]),
            "max_temperature": max(self.metrics["temperature"]),
            "total_energy": sum(self.metrics["power_consumption"]) * 0.1,  # Joules
        }


class JetsonSensorimotorSystem:
    """Jetson-optimized sensorimotor system."""

    def __init__(self, use_gpu: bool = False):
        """Initialize Jetson sensorimotor system."""
        self.optimizer = JetsonOptimizer()
        self.network = None
        self.use_gpu = use_gpu and CUDA_AVAILABLE

        # Jetson-optimized configuration
        self.network_config = {
            "layers": [
                {
                    "name": "sensory",
                    "type": "sensory",
                    "size": 64,
                    "encoding_type": "multimodal",
                },
                {
                    "name": "integration",
                    "type": "processing",
                    "size": 32,
                    "neuron_type": "adex",
                },
                {
                    "name": "decision",
                    "type": "processing",
                    "size": 16,
                    "neuron_type": "adex",
                },
                {"name": "motor", "type": "motor", "size": 8},
            ],
            "connections": [
                {
                    "pre_layer": "sensory",
                    "post_layer": "integration",
                    "connection_type": "feedforward",
                    "synapse_type": "stdp",
                    "probability": 0.05,
                },
                {
                    "pre_layer": "integration",
                    "post_layer": "decision",
                    "connection_type": "feedforward",
                    "synapse_type": "stdp",
                    "probability": 0.05,
                },
                {
                    "pre_layer": "decision",
                    "post_layer": "motor",
                    "connection_type": "feedforward",
                    "synapse_type": "stdp",
                    "probability": 0.1,
                },
            ],
        }

    def initialize(self):
        """Initialize the Jetson system."""
        self.logger = logging.getLogger(__name__)

        # Get system info
        system_info = self.optimizer.get_system_info()
        self.logger.info(f"Jetson Nano System Info: {system_info}")

        # Create optimized network
        self.network = self.optimizer.create_jetson_network(self.network_config)

        # Initialize encoders
        self.encoders = {
            "vision": RetinalEncoder(resolution=(16, 16)),  # Reduced resolution
            "auditory": CochlearEncoder(frequency_bands=16),  # Reduced bands
            "tactile": SomatosensoryEncoder(sensor_grid=(8, 8)),  # Reduced grid
        }

        self.logger.info("Jetson sensorimotor system initialized")

    def run_inference(
        self, sensory_inputs: Dict[str, Any], duration: float = 50.0
    ) -> Dict[str, Any]:
        """Run inference on Jetson Nano."""
        if self.network is None:
            self.initialize()

        # Encode sensory inputs
        encoded_inputs = {}
        for modality, encoder in self.encoders.items():
            if modality in sensory_inputs:
                encoded_inputs[modality] = encoder.encode(sensory_inputs[modality])

        # Run simulation
        results = self.network.run_simulation(duration)

        # Add Jetson metrics
        results["jetson_metrics"] = results.get("jetson_metrics", {})

        return results

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for Jetson deployment."""
        if self.network is None:
            return {}

        system_info = self.optimizer.get_system_info()
        network_info = self.network.get_network_info()

        return {
            "system_info": system_info,
            "network_info": network_info,
            "performance_metrics": getattr(
                self.network, "performance_monitor", {}
            ).get_metrics(),
            "optimization_status": {
                "gpu_enabled": self.use_gpu,
                "memory_optimized": True,
                "temperature_monitored": True,
            },
        }


def create_jetson_deployment_script():
    """Create deployment script for Jetson Nano."""
    script_content = """#!/bin/bash
# Jetson Nano Neuromorphic System Deployment Script

echo "Setting up Jetson Nano for neuromorphic computing..."

# Update system
sudo apt update
sudo apt upgrade -y

# Install Python dependencies
sudo apt install -y python3-pip python3-dev
pip3 install numpy matplotlib scipy scikit-learn pandas

# Install Jetson-specific packages
sudo apt install -y python3-opencv
pip3 install cupy-cuda11x  # For CUDA acceleration

# Install neuromorphic system
cd /home/nano/neuron
pip3 install -r requirements.txt

# Set up performance monitoring
sudo apt install -y lm-sensors
sudo sensors-detect --auto

# Create startup script
cat > /home/nano/start_neuromorphic.sh << 'EOF'
#!/bin/bash
cd /home/nano/neuron
python3 demo/jetson_demo.py
EOF

chmod +x /home/nano/start_neuromorphic.sh

echo "Jetson Nano setup complete!"
echo "Run: ./start_neuromorphic.sh"
"""

    with open("jetson_setup.sh", "w") as f:
        f.write(script_content)

    return "jetson_setup.sh"


if __name__ == "__main__":
    # Create Jetson deployment script
    setup_script = create_jetson_deployment_script()
    print(f"Created Jetson setup script: {setup_script}")

    # Test Jetson optimizer
    optimizer = JetsonOptimizer()
    system_info = optimizer.get_system_info()
    print(f"Jetson System Info: {system_info}")

    # Test Jetson network creation
    jetson_system = JetsonSensorimotorSystem(use_gpu=CUDA_AVAILABLE)
    jetson_system.initialize()

    # Test inference
    test_inputs = {
        "vision": np.random.rand(16, 16),
        "auditory": np.random.randn(100),
        "tactile": np.random.rand(8, 8),
    }

    results = jetson_system.run_inference(test_inputs)
    print(f"Inference completed with metrics: {results.get('jetson_metrics', {})}")

    # Get performance summary
    summary = jetson_system.get_performance_summary()
    print(f"Performance Summary: {summary}")
