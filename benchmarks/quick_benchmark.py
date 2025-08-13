"""
Quick Performance Benchmarks
=============================

A simplified benchmark suite for rapid testing with smaller networks.
"""

import os
import sys
import csv
import time
import psutil
import argparse
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.network import NeuromorphicNetwork
from core.neurons import NeuronPopulation


def run_network_benchmark(
    num_neurons,
    sim_time_ms=100,
    dt=1.0,
    strong_input=False,
    conn_prob=0.02,
    max_seconds=None,
):
    """Run a simple network benchmark."""
    
    # Build network
    network = NeuromorphicNetwork()
    
    # Simple 2-layer network
    input_size = num_neurons // 2
    output_size = num_neurons - input_size
    
    network.add_layer("input", input_size, "lif")  # Use simpler LIF neurons
    network.add_layer("output", output_size, "lif")
    
    # Sparse connections
    network.connect_layers("input", "output", "stdp", connection_probability=float(conn_prob))
    
    # Measure memory before
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # Run simulation
    start_time = time.perf_counter()
    
    num_steps = int(sim_time_ms / dt)
    total_spikes = 0
    
    wall_start = time.perf_counter()
    for step in range(num_steps):
        # Inject input
        input_layer = network.layers["input"]
        if strong_input:
            # Sustained, stronger drive: stimulate a small fraction every step
            stim_count = max(10, min(200, int(input_size * 0.01)))
            for neuron in input_layer.neuron_population.neurons[:stim_count]:
                neuron.membrane_potential += np.random.uniform(10, 20)
        else:
            # Light, intermittent drive every 20 ms to first 10 neurons
            if step % 20 == 0:
                for neuron in input_layer.neuron_population.neurons[:10]:
                    neuron.membrane_potential += np.random.uniform(5, 10)
        
        # Step network
        network.step(dt)
        
        # Count spikes
        for layer in network.layers.values():
            for neuron in layer.neuron_population.neurons:
                if hasattr(neuron, 'is_spiking') and neuron.is_spiking:
                    total_spikes += 1
                    neuron.is_spiking = False

        # Optional wall-clock cap for large sizes
        if max_seconds is not None and (time.perf_counter() - wall_start) > float(max_seconds):
            break
    
    elapsed_time = time.perf_counter() - start_time
    
    # Measure memory after
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    memory_used = mem_after - mem_before
    
    # Calculate metrics
    spike_rate = total_spikes / (sim_time_ms / 1000.0) / num_neurons if num_neurons > 0 else 0
    throughput = total_spikes / elapsed_time if elapsed_time > 0 else 0
    
    return {
        'neurons': num_neurons,
        'time_s': elapsed_time,
        'memory_mb': memory_used,
        'total_spikes': total_spikes,
        'spike_rate_hz': spike_rate,
        'throughput': throughput,
        'steps': num_steps
    }


def main():
    """Run quick benchmarks."""
    parser = argparse.ArgumentParser(description="Quick neuromorphic benchmarks")
    parser.add_argument(
        "--sizes",
        type=str,
        default="100,500,1000,5000",
        help="Comma-separated neuron sizes to benchmark",
    )
    parser.add_argument(
        "--strong-input",
        action="store_true",
        help="Use stronger/sustained input drive for larger sizes",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Wall-clock cap per size (seconds) to keep runs short",
    )
    parser.add_argument(
        "--conn-prob",
        type=float,
        default=0.02,
        help="Connection probability for input->output",
    )
    args = parser.parse_args()
    print("="*60)
    print("Quick Performance Benchmarks")
    print("="*60)
    
    # Test configurations
    try:
        network_sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    except Exception:
        network_sizes = [100, 500, 1000, 5000]
    results = []
    
    for size in network_sizes:
        print(f"\nBenchmarking {size} neurons...")
        
        # Adjust simulation time based on size
        if size <= 500:
            sim_time = 200
        elif size <= 1000:
            sim_time = 100
        elif size <= 5000:
            sim_time = 50
        else:
            sim_time = 30  # cap for very large sizes to keep runtime short
        
        try:
            # For very large sizes, modestly reduce connection probability
            conn_prob = args.conn_prob if size <= 5000 else min(args.conn_prob, 0.01)
            result = run_network_benchmark(
                size,
                sim_time,
                strong_input=args.strong_input,
                conn_prob=conn_prob,
                max_seconds=args.max_seconds,
            )
            results.append(result)
            
            print(f"  ✓ Time: {result['time_s']:.2f}s")
            print(f"    Memory: {result['memory_mb']:.1f} MB")
            print(f"    Spikes: {result['total_spikes']}")
            print(f"    Rate: {result['spike_rate_hz']:.1f} Hz/neuron")
            print(f"    Throughput: {result['throughput']:.0f} spikes/s")
            
        except Exception as e:
            print(f"  ✗ Failed: {str(e)}")
            results.append({
                'neurons': size,
                'time_s': 0,
                'memory_mb': 0,
                'total_spikes': 0,
                'spike_rate_hz': 0,
                'throughput': 0,
                'steps': 0,
                'error': str(e)
            })
    
    # Save results
    os.makedirs("benchmark_results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"benchmark_results/quick_benchmark_{timestamp}.csv"
    
    with open(csv_file, 'w', newline='') as f:
        fieldnames = ['neurons', 'time_s', 'memory_mb', 'total_spikes', 
                     'spike_rate_hz', 'throughput', 'steps']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            # Skip error field when writing
            row = {k: v for k, v in result.items() if k != 'error'}
            writer.writerow(row)
    
    print(f"\nResults saved to: {csv_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"{'Size':<10} {'Time(s)':<10} {'Memory(MB)':<12} {'Throughput':<15}")
    print("-"*60)
    for result in results:
        if 'error' not in result:
            print(f"{result['neurons']:<10} {result['time_s']:<10.2f} "
                  f"{result['memory_mb']:<12.1f} {result['throughput']:<15.0f}")
    
    # Test GPU if available
    try:
        import torch
        if torch.cuda.is_available():
            print("\n" + "="*60)
            print("GPU Information")
            print("="*60)
            print(f"CUDA Available: {torch.cuda.is_available()}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    except ImportError:
        pass
    
    try:
        import cupy as cp
        print("\n" + "="*60)
        print("CuPy Available")
        print("="*60)
        print(f"Version: {cp.__version__}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
