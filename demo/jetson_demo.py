"""
Jetson Nano Neuromorphic System Demo
====================================

This demo showcases the neuromorphic programming system running on NVIDIA Jetson Nano 8GB
for edge neuromorphic computing applications.
"""

from core.encoding import CochlearEncoder, RetinalEncoder, SomatosensoryEncoder
from jetson_optimization import JetsonOptimizer, JetsonSensorimotorSystem
import logging
import os
import sys
import time
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demonstrate_jetson_system_info():
    """Demonstrate Jetson Nano system information."""
    print("\n=== Jetson Nano System Information ===")

    optimizer = JetsonOptimizer()
    system_info = optimizer.get_system_info()

    print(f"CPU Cores: {system_info['cpu_count']}")
    print(f"Total Memory: {system_info['memory_total'] / (1024**3):.2f} GB")
    print(f"Available Memory: {system_info['memory_available'] / (1024**3):.2f} GB")
    print(f"CUDA Available: {system_info['cuda_available']}")
    print(f"OpenCV Available: {system_info['opencv_available']}")
    print(f"Temperature: {system_info['temperature']:.1f}°C")
    print(f"Power Consumption: {system_info['power_consumption']:.2f}W")

    return system_info


def demonstrate_jetson_optimization():
    """Demonstrate Jetson optimization capabilities."""
    print("\n=== Jetson Optimization Demonstration ===")

    optimizer = JetsonOptimizer()

    # Test different network sizes
    test_configs = [
        {"neurons": 500, "synapses": 5000},
        {"neurons": 1000, "synapses": 10000},
        {"neurons": 2000, "synapses": 20000},
    ]

    for config in test_configs:
        print(
            f"\nTesting network: {config['neurons']} neurons, {config['synapses']} synapses"
        )
        optimization = optimizer.optimize_network_size(
            target_neurons=config["neurons"], target_synapses=config["synapses"]
        )
        print(
            f"Optimized to: {optimization['max_neurons']} neurons, {optimization['max_synapses']} synapses"
        )
        print(f"Recommended layers: {optimization['recommended_layers']}")


def demonstrate_jetson_inference():
    """Demonstrate real-time inference on Jetson Nano."""
    print("\n=== Jetson Inference Demonstration ===")

    # Initialize Jetson system
    jetson_system = JetsonSensorimotorSystem(use_gpu=True)
    jetson_system.initialize()

    # Create test inputs
    test_inputs = {
        "vision": np.random.rand(16, 16),
        "auditory": np.random.randn(100),
        "tactile": np.random.rand(8, 8),
    }

    # Run multiple inference cycles
    inference_times = []
    performance_metrics = []

    for i in range(10):
        print(f"Inference cycle {i+1}/10...")

        start_time = time.time()
        results = jetson_system.run_inference(test_inputs, duration=50.0)
        inference_time = time.time() - start_time

        inference_times.append(inference_time)
        performance_metrics.append(results.get("jetson_metrics", {}))

        print(f"  Inference time: {inference_time:.3f}s")
        if "jetson_metrics" in results:
            metrics = results["jetson_metrics"]
            print(f"  CPU: {metrics.get('current_cpu', 0):.1f}%")
            print(f"  Memory: {metrics.get('current_memory', 0):.1f}%")
            print(f"  Temperature: {metrics.get('current_temperature', 0):.1f}°C")
            print(f"  Power: {metrics.get('current_power', 0):.2f}W")

    # Calculate statistics
    avg_inference_time = np.mean(inference_times)
    avg_cpu = np.mean([m.get("current_cpu", 0) for m in performance_metrics])
    avg_memory = np.mean([m.get("current_memory", 0) for m in performance_metrics])
    max_temp = max([m.get("current_temperature", 0) for m in performance_metrics])

    print(f"\nInference Statistics:")
    print(f"  Average inference time: {avg_inference_time:.3f}s")
    print(f"  Average CPU usage: {avg_cpu:.1f}%")
    print(f"  Average memory usage: {avg_memory:.1f}%")
    print(f"  Maximum temperature: {max_temp:.1f}°C")

    return {
        "inference_times": inference_times,
        "performance_metrics": performance_metrics,
        "statistics": {
            "avg_inference_time": avg_inference_time,
            "avg_cpu": avg_cpu,
            "avg_memory": avg_memory,
            "max_temp": max_temp,
        },
    }


def demonstrate_jetson_learning():
    """Demonstrate learning capabilities on Jetson Nano."""
    print("\n=== Jetson Learning Demonstration ===")

    # Initialize Jetson system
    jetson_system = JetsonSensorimotorSystem(use_gpu=True)
    jetson_system.initialize()

    # Create training data
    training_data = []
    for i in range(20):
        training_data.append(
            {
                "vision": np.random.rand(16, 16),
                "auditory": np.random.randn(100),
                "tactile": np.random.rand(8, 8),
                "target": i % 8,  # 8 motor neurons
            }
        )

    # Run learning trials
    learning_metrics = []

    for epoch in range(5):
        print(f"Learning epoch {epoch+1}/5...")
        epoch_metrics = []

        for trial in training_data:
            start_time = time.time()
            results = jetson_system.run_inference(trial, duration=50.0)
            inference_time = time.time() - start_time

            # Calculate reward (simplified)
            motor_spikes = results.get("layer_spike_times", {}).get("motor", [])
            action = len(motor_spikes) if motor_spikes else 0
            reward = 1.0 if action == trial["target"] else -0.2

            epoch_metrics.append(
                {
                    "inference_time": inference_time,
                    "reward": reward,
                    "performance": results.get("jetson_metrics", {}),
                }
            )

        # Calculate epoch statistics
        avg_reward = np.mean([m["reward"] for m in epoch_metrics])
        avg_inference_time = np.mean([m["inference_time"] for m in epoch_metrics])
        avg_cpu = np.mean(
            [m["performance"].get("current_cpu", 0) for m in epoch_metrics]
        )

        learning_metrics.append(
            {
                "epoch": epoch,
                "avg_reward": avg_reward,
                "avg_inference_time": avg_inference_time,
                "avg_cpu": avg_cpu,
            }
        )

        print(f"  Average reward: {avg_reward:.3f}")
        print(f"  Average inference time: {avg_inference_time:.3f}s")
        print(f"  Average CPU usage: {avg_cpu:.1f}%")

    return learning_metrics


def demonstrate_jetson_performance_monitoring():
    """Demonstrate performance monitoring on Jetson Nano."""
    print("\n=== Jetson Performance Monitoring ===")

    jetson_system = JetsonSensorimotorSystem(use_gpu=True)
    jetson_system.initialize()

    # Get performance summary
    summary = jetson_system.get_performance_summary()

    print("Performance Summary:")
    print(f"  System Info: {summary.get('system_info', {})}")
    print(f"  Network Info: {summary.get('network_info', {})}")
    print(f"  Optimization Status: {summary.get('optimization_status', {})}")

    # Monitor over time
    print("\nReal-time monitoring (10 seconds)...")
    monitoring_data = []

    for i in range(10):
        time.sleep(1)

        # Run a quick inference
        test_inputs = {
            "vision": np.random.rand(16, 16),
            "auditory": np.random.randn(100),
            "tactile": np.random.rand(8, 8),
        }

        results = jetson_system.run_inference(test_inputs, duration=10.0)
        metrics = results.get("jetson_metrics", {})

        monitoring_data.append(
            {
                "timestamp": time.time(),
                "cpu": metrics.get("current_cpu", 0),
                "memory": metrics.get("current_memory", 0),
                "temperature": metrics.get("current_temperature", 0),
                "power": metrics.get("current_power", 0),
            }
        )

        print(
            f"  {i+1}s: CPU={metrics.get('current_cpu', 0):.1f}%, "
            f"Mem={metrics.get('current_memory', 0):.1f}%, "
            f"Temp={metrics.get('current_temperature', 0):.1f}°C, "
            f"Power={metrics.get('current_power', 0):.2f}W"
        )

    return monitoring_data


def plot_jetson_performance(monitoring_data):
    """Plot Jetson performance data."""
    if not monitoring_data:
        return

    timestamps = [
        d["timestamp"] - monitoring_data[0]["timestamp"] for d in monitoring_data
    ]
    cpu_usage = [d["cpu"] for d in monitoring_data]
    memory_usage = [d["memory"] for d in monitoring_data]
    temperature = [d["temperature"] for d in monitoring_data]
    power_consumption = [d["power"] for d in monitoring_data]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

    # CPU Usage
    ax1.plot(timestamps, cpu_usage, "b-", linewidth=2)
    ax1.set_title("CPU Usage")
    ax1.set_ylabel("CPU (%)")
    ax1.grid(True, alpha=0.3)

    # Memory Usage
    ax2.plot(timestamps, memory_usage, "g-", linewidth=2)
    ax2.set_title("Memory Usage")
    ax2.set_ylabel("Memory (%)")
    ax2.grid(True, alpha=0.3)

    # Temperature
    ax3.plot(timestamps, temperature, "r-", linewidth=2)
    ax3.set_title("Temperature")
    ax3.set_ylabel("Temperature (°C)")
    ax3.grid(True, alpha=0.3)

    # Power Consumption
    ax4.plot(timestamps, power_consumption, "m-", linewidth=2)
    ax4.set_title("Power Consumption")
    ax4.set_ylabel("Power (W)")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("jetson_performance.png", dpi=300, bbox_inches="tight")
    print("Performance plot saved as 'jetson_performance.png'")


def main():
    """Run Jetson Nano neuromorphic system demonstration."""
    print("Jetson Nano Neuromorphic System Demonstration")
    print("=" * 50)

    try:
        # System information
        system_info = demonstrate_jetson_system_info()

        # Optimization demonstration
        demonstrate_jetson_optimization()

        # Inference demonstration
        inference_results = demonstrate_jetson_inference()

        # Learning demonstration
        learning_results = demonstrate_jetson_learning()

        # Performance monitoring
        monitoring_data = demonstrate_jetson_performance_monitoring()

        # Plot performance
        plot_jetson_performance(monitoring_data)

        print("\n=== Jetson Demo Complete ===")
        print("All demonstrations completed successfully!")

        # Print summary
        print(f"\nSummary:")
        print(f"  System: Jetson Nano 8GB")
        print(f"  CUDA: {system_info['cuda_available']}")
        print(
            f"  Average inference time: {inference_results['statistics']['avg_inference_time']:.3f}s"
        )
        print(f"  Average CPU usage: {inference_results['statistics']['avg_cpu']:.1f}%")
        print(
            f"  Maximum temperature: {inference_results['statistics']['max_temp']:.1f}°C"
        )

    except Exception as e:
        print(f"Error during Jetson demonstration: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
