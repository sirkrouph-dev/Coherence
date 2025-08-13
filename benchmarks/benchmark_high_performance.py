"""
Comprehensive benchmark for the new high-performance vectorized architecture.
Tests large-scale network performance and scalability.
"""

import numpy as np
import time
import sys
import os
from typing import List, Dict, Any

# Add the core directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

from core.high_performance_network import HighPerformanceNeuromorphicNetwork
from core.vectorized_neurons import create_vectorized_population
from core.vectorized_synapses import VectorizedSynapseManager


def create_feedforward_network(layer_sizes: List[int], plasticity: bool = True):
    """Helper function to create feedforward networks for testing."""
    network = HighPerformanceNeuromorphicNetwork()
    
    # Add layers
    for i, size in enumerate(layer_sizes):
        layer_name = f"layer_{i}"
        network.add_layer(layer_name, size, "lif")
    
    # Connect adjacent layers
    for i in range(len(layer_sizes) - 1):
        layer_name = f"layer_{i}"
        next_layer = f"layer_{i+1}"
        synapse_type = "stdp" if plasticity else "static"
        
        # Use sparse connectivity to avoid memory explosion
        total_neurons = sum(layer_sizes)
        if total_neurons > 10000:
            connectivity = 0.01  # 1% for large networks
        elif total_neurons > 3000:
            connectivity = 0.02  # 2% for medium networks
        else:
            connectivity = 0.1   # 10% for small networks
            
        network.connect_layers(layer_name, next_layer, synapse_type, connectivity)
    
    return network


def benchmark_vectorized_architecture():
    """Comprehensive benchmark of the new vectorized architecture."""
    print("=" * 80)
    print("HIGH-PERFORMANCE VECTORIZED ARCHITECTURE BENCHMARK")
    print("=" * 80)
    
    # Test 1: Vectorized Neuron Performance
    print("\n1. VECTORIZED NEURON PERFORMANCE")
    print("-" * 50)
    benchmark_vectorized_neurons()
    
    # Test 2: Vectorized Synapse Performance
    print("\n2. VECTORIZED SYNAPSE PERFORMANCE")
    print("-" * 50)
    benchmark_vectorized_synapses()
    
    # Test 3: Full Network Performance
    print("\n3. FULL NETWORK PERFORMANCE")
    print("-" * 50)
    benchmark_full_networks()
    
    # Test 4: Large Scale Performance
    print("\n4. LARGE SCALE PERFORMANCE")
    print("-" * 50)
    benchmark_large_scale()
    
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


def benchmark_vectorized_neurons():
    """Test vectorized neuron performance."""
    sizes = [100, 500, 1000, 5000, 10000]
    
    print(f"{'Size':<8} {'Creation':<12} {'Simulation':<12} {'Throughput':<15} {'Memory':<10}")
    print("-" * 65)
    
    for size in sizes:
        # Creation time
        start_time = time.time()
        pop = create_vectorized_population(size, "lif")
        creation_time = time.time() - start_time
        
        # Simulation time
        I_syn = np.random.uniform(80, 120, size)  # Strong currents
        num_steps = 100  # 10ms simulation
        
        start_time = time.time()
        total_spikes = 0
        for _ in range(num_steps):
            spikes = pop.step(0.1, I_syn)
            total_spikes += np.sum(spikes)
        simulation_time = time.time() - start_time
        
        # Calculate throughput
        neuron_steps = size * num_steps
        throughput = neuron_steps / simulation_time if simulation_time > 0 else 0
        
        # Memory estimate
        memory_mb = (size * 4 * 6) / (1024 * 1024)  # 6 float32 arrays
        
        print(f"{size:<8} {creation_time*1000:<10.2f}ms {simulation_time*1000:<10.2f}ms "
              f"{throughput/1000:<12.1f}k/s {memory_mb:<8.2f}MB")


def benchmark_vectorized_synapses():
    """Test vectorized synapse performance."""
    test_cases = [
        (100, 100, 0.1),
        (500, 500, 0.1),
        (1000, 1000, 0.1),
        (1000, 1000, 0.2),
        (2000, 2000, 0.1)
    ]
    
    print(f"{'Pre x Post':<12} {'Density':<8} {'Synapses':<10} {'Creation':<12} {'Compute':<12} {'Update':<12}")
    print("-" * 80)
    
    for pre_size, post_size, density in test_cases:
        # Create synapse manager
        manager = VectorizedSynapseManager()
        manager.add_layer("pre", pre_size)
        manager.add_layer("post", post_size)
        
        # Creation time
        start_time = time.time()
        manager.connect_layers("pre", "post", "stdp", density)
        creation_time = time.time() - start_time
        
        # Get number of synapses
        stats = manager.get_all_statistics()
        n_synapses = stats["total_connections"]
        
        # Test current computation
        pre_spikes = np.random.random(pre_size) < 0.1  # 10% spike probability
        post_spikes = np.random.random(post_size) < 0.1
        
        start_time = time.time()
        for _ in range(50):  # 50 steps
            currents = manager.compute_layer_currents({"pre": pre_spikes}, 0.0)
        compute_time = time.time() - start_time
        
        # Test weight updates
        start_time = time.time()
        for _ in range(50):  # 50 steps
            manager.update_all_weights({"pre": pre_spikes, "post": post_spikes}, 0.0)
        update_time = time.time() - start_time
        
        print(f"{pre_size}x{post_size:<6} {density:<8.1f} {n_synapses:<10} "
              f"{creation_time*1000:<10.2f}ms {compute_time*1000:<10.2f}ms {update_time*1000:<10.2f}ms")


def benchmark_full_networks():
    """Test complete network performance."""
    network_configs = [
        ("Small", [100, 50, 20]),
        ("Medium", [500, 200, 100]),
        ("Large", [1000, 500, 200]),
        ("XLarge", [2000, 1000, 500])
    ]
    
    print(f"{'Network':<8} {'Neurons':<8} {'Creation':<12} {'Simulation':<12} {'Spikes/s':<12} {'Memory':<10}")
    print("-" * 75)
    
    for name, layer_sizes in network_configs:
        total_neurons = sum(layer_sizes)
        
        # Creation time
        start_time = time.time()
        net = create_feedforward_network(layer_sizes, plasticity=True)
        creation_time = time.time() - start_time
        
        # Simulation performance
        start_memory = get_memory_mb()
        start_time = time.time()
        
        results = net.run_simulation(duration=10.0, dt=0.1)  # 10ms simulation
        
        simulation_time = time.time() - start_time
        end_memory = get_memory_mb()
        
        # Extract metrics
        total_spikes = results["total_spikes"]
        spikes_per_second = total_spikes / (results["simulation_time"])
        memory_used = end_memory - start_memory
        
        print(f"{name:<8} {total_neurons:<8} {creation_time*1000:<10.2f}ms "
              f"{simulation_time*1000:<10.2f}ms {spikes_per_second:<10.1f} {memory_used:<8.2f}MB")


def benchmark_large_scale():
    """Test large-scale network performance."""
    print("Testing large-scale networks (may take time)...")
    print()
    
    large_configs = [
        ("2K", [2000, 1000]),
        ("5K", [5000, 2000]),
        ("10K", [10000, 5000]),
        ("15K", [15000, 7500])
    ]
    
    print(f"{'Size':<6} {'Neurons':<8} {'Synapses':<10} {'Creation':<12} {'Step Time':<12} {'Throughput':<15}")
    print("-" * 80)
    
    for name, layer_sizes in large_configs:
        try:
            total_neurons = sum(layer_sizes)
            
            # Creation
            start_time = time.time()
            net = create_feedforward_network(layer_sizes, plasticity=False)  # No plasticity for speed
            creation_time = time.time() - start_time
            
            # Get network info
            info = net.get_network_info()
            total_synapses = info["synapse_statistics"]["total_connections"]
            
            # Test a few steps to measure performance
            num_test_steps = 10
            net.reset()
            
            start_time = time.time()
            for _ in range(num_test_steps):
                net.step(0.1)
            step_time = (time.time() - start_time) / num_test_steps
            
            # Calculate throughput
            throughput = (total_neurons * 1000) / step_time if step_time > 0 else 0  # neurons/second
            
            print(f"{name:<6} {total_neurons:<8} {total_synapses:<10} "
                  f"{creation_time:<10.3f}s {step_time*1000:<10.2f}ms {throughput/1000:<12.1f}k/s")
            
        except Exception as e:
            print(f"{name:<6} FAILED: {str(e)}")


def get_memory_mb() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def test_basic_functionality():
    """Test basic functionality of the new architecture."""
    print("Testing basic functionality...")
    
    # Test 1: Create simple network
    print("✓ Creating simple network...")
    net = create_feedforward_network([100, 50, 20])
    
    # Test 2: Run simulation
    print("✓ Running simulation...")
    results = net.run_simulation(duration=5.0, dt=0.1)
    
    # Test 3: Check results
    print("✓ Checking results...")
    assert results["duration"] == 5.0
    assert "simulation_time" in results
    assert len(results["layer_spike_counts"]) == 3
    
    # Test 4: Network info
    print("✓ Getting network info...")
    info = net.get_network_info()
    assert info["total_neurons"] == 170
    assert info["synapse_statistics"]["total_connections"] > 0
    
    print("✓ All functionality tests passed!")


if __name__ == "__main__":
    # Test basic functionality first
    try:
        test_basic_functionality()
        print()
        
        # Run full benchmark
        benchmark_vectorized_architecture()
        
    except Exception as e:
        print(f"Error during benchmarking: {e}")
        import traceback
        traceback.print_exc()
