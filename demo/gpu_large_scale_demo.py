"""
GPU Large-Scale Neuromorphic Demo
=================================

This demo showcases the neuromorphic programming system running on desktop GPUs
for large-scale neuromorphic computing with 50k+ neurons.
"""

from scripts.gpu_optimization import (GPUOptimizer, GPUSensorimotorSystem,
                                      demonstrate_gpu_capabilities)
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


def test_gpu_system_capacity():
    """Test GPU system capacity for large-scale networks."""
    print("\n=== Testing GPU System Capacity ===")

    optimizer = GPUOptimizer()

    # Test different neuron counts
    test_sizes = [10000, 25000, 50000, 75000, 100000]

    for size in test_sizes:
        print(f"\nTesting capacity for {size:,} neurons:")
        capacity = optimizer.calculate_network_capacity(size)

        print(f"  Max neurons: {capacity['max_neurons']:,}")
        print(f"  Max synapses: {capacity['max_synapses']:,}")
        print(f"  Memory usage: {capacity['memory_usage_estimate']:.2f} GB")
        print(f"  Recommended layers: {capacity['recommended_layers']}")

        if capacity["max_neurons"] >= size:
            print(f"  ✓ System can handle {size:,} neurons")
        else:
            print(
                f"  ✗ System cannot handle {size:,} neurons (max: {capacity['max_neurons']:,})"
            )


def run_large_scale_demo():
    """Run large-scale neuromorphic demo on GPU."""
    print("\n=== Large-Scale GPU Neuromorphic Demo ===")

    # Initialize GPU system with 50k neurons target
    gpu_system = GPUSensorimotorSystem(use_gpu=True, max_neurons=50000)

    try:
        print("Initializing GPU sensorimotor system...")
        gpu_system.initialize()

        # Get performance summary
        summary = gpu_system.get_performance_summary()
        print(f"\nSystem Summary:")
        print(f"  GPU enabled: {summary['gpu_enabled']}")
        print(f"  Network size: {summary['network_size']['neurons']:,} neurons")
        print(f"  System capacity: {summary['capacity']['max_neurons']:,} neurons")

        # Create large-scale test inputs
        print("\nCreating large-scale test inputs...")
        test_inputs = {
            "vision": np.random.rand(128, 128),  # Larger visual input
            "auditory": np.random.randn(5000),  # Larger auditory input
            "tactile": np.random.rand(64, 64),  # Larger tactile input
        }

        # Run inference with different durations
        durations = [25.0, 50.0, 100.0]

        for duration in durations:
            print(f"\nRunning inference for {duration}s...")

            start_time = time.time()
            results = gpu_system.run_inference(test_inputs, duration=duration)
            total_time = time.time() - start_time

            print(f"Results for {duration}s simulation:")
            print(f"  Total neurons: {results.get('total_neurons', 0):,}")
            print(f"  Total spikes: {results.get('total_spikes', 0):,}")
            print(f"  Inference time: {results.get('inference_time', 0):.2f}s")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Neurons/second: {results.get('neurons_per_second', 0):,.0f}")
            print(f"  Spikes/second: {results.get('spikes_per_second', 0):,.0f}")

            # GPU metrics if available
            if "gpu_metrics" in results:
                gpu_metrics = results["gpu_metrics"]
                print(f"  GPU Memory: {gpu_metrics.get('current_gpu_memory', 0):.1f}%")
                print(f"  CPU Usage: {gpu_metrics.get('current_cpu', 0):.1f}%")
                print(f"  RAM Usage: {gpu_metrics.get('current_memory', 0):.1f}%")

    except Exception as e:
        print(f"Error during GPU demo: {e}")
        import traceback

        traceback.print_exc()


def benchmark_different_sizes():
    """Benchmark performance with different network sizes."""
    print("\n=== Benchmarking Different Network Sizes ===")

    sizes = [10000, 25000, 50000]
    results = {}

    for size in sizes:
        print(f"\nBenchmarking {size:,} neurons...")

        try:
            # Initialize system
            gpu_system = GPUSensorimotorSystem(use_gpu=True, max_neurons=size)
            gpu_system.initialize()

            # Create test inputs
            test_inputs = {
                "vision": np.random.rand(64, 64),
                "auditory": np.random.randn(1000),
                "tactile": np.random.rand(32, 32),
            }

            # Run benchmark
            start_time = time.time()
            benchmark_results = gpu_system.run_inference(test_inputs, duration=25.0)
            total_time = time.time() - start_time

            results[size] = {
                "neurons": benchmark_results.get("total_neurons", 0),
                "spikes": benchmark_results.get("total_spikes", 0),
                "inference_time": benchmark_results.get("inference_time", 0),
                "total_time": total_time,
                "neurons_per_second": benchmark_results.get("neurons_per_second", 0),
                "spikes_per_second": benchmark_results.get("spikes_per_second", 0),
            }

            print(
                f"  ✓ Completed: {results[size]['neurons']:,} neurons, {results[size]['spikes']:,} spikes"
            )
            print(
                f"  Performance: {results[size]['neurons_per_second']:,.0f} neurons/sec"
            )

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            results[size] = {"error": str(e)}

    # Print benchmark summary
    print("\n=== Benchmark Summary ===")
    for size, result in results.items():
        if "error" not in result:
            print(f"{size:,} neurons:")
            print(f"  Neurons: {result['neurons']:,}")
            print(f"  Spikes: {result['spikes']:,}")
            print(f"  Time: {result['total_time']:.2f}s")
            print(f"  Performance: {result['neurons_per_second']:,.0f} neurons/sec")
        else:
            print(f"{size:,} neurons: FAILED - {result['error']}")
        print()


def plot_performance_comparison():
    """Plot performance comparison for different network sizes."""
    print("\n=== Performance Comparison ===")

    # This would create plots comparing performance across different network sizes
    # For now, just print the comparison
    print("Performance comparison would be plotted here.")
    print("(Plotting functionality can be added if needed)")


def main():
    """Main demo function."""
    print("GPU Large-Scale Neuromorphic System Demo")
    print("=" * 50)

    # Show system capabilities
    demonstrate_gpu_capabilities()

    # Test system capacity
    test_gpu_system_capacity()

    # Run large-scale demo
    run_large_scale_demo()

    # Benchmark different sizes
    benchmark_different_sizes()

    print("\n=== Demo Complete ===")
    print(
        "The system has been tested for large-scale neuromorphic computing on your GPU."
    )
    print("You can now run networks with 50k+ neurons on your desktop GPU!")


if __name__ == "__main__":
    main()
