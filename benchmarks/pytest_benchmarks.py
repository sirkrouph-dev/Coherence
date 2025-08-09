"""
Performance Benchmarks using pytest-benchmark
=============================================

This module implements comprehensive performance benchmarks using pytest-benchmark
for accurate and reproducible measurements of:
- Step throughput (neurons/sec)
- Memory footprint
- Convergence speed on standard tasks
"""

import gc
import psutil
import numpy as np
import pytest
from typing import Dict, List, Tuple
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.network import Network
from engine.neuron_group import NeuronGroup
from engine.synapse_group import SynapseGroup
from engine.simulator import Simulator, SimulationMode
from core.neurons import LeakyIntegrateAndFire, AdaptiveExponentialIntegrateAndFire


# -----------------------------------------------------------------------------
# Benchmark Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture(params=[100, 1000, 5000, 10000])
def network_size(request):
    """Network size parameter for benchmarks."""
    return request.param


@pytest.fixture(params=['lif', 'adex'])
def neuron_model(request):
    """Neuron model type for benchmarks."""
    return request.param


@pytest.fixture
def small_network():
    """Create a small test network."""
    network = Network("small_benchmark")
    network.add_neuron_group("input", 100, "lif")
    network.add_neuron_group("hidden", 100, "lif")
    network.add_neuron_group("output", 10, "lif")
    
    network.connect("input", "hidden", connection_probability=0.1)
    network.connect("hidden", "output", connection_probability=0.1)
    
    return network


@pytest.fixture
def large_network():
    """Create a large test network."""
    network = Network("large_benchmark")
    network.add_neuron_group("input", 1000, "lif")
    network.add_neuron_group("hidden1", 500, "lif")
    network.add_neuron_group("hidden2", 500, "lif")
    network.add_neuron_group("output", 100, "lif")
    
    network.connect("input", "hidden1", connection_probability=0.05)
    network.connect("hidden1", "hidden2", connection_probability=0.05)
    network.connect("hidden2", "output", connection_probability=0.05)
    
    return network


# -----------------------------------------------------------------------------
# Step Throughput Benchmarks
# -----------------------------------------------------------------------------

class TestStepThroughput:
    """Benchmarks measuring neuron processing throughput."""
    
    def test_single_neuron_step(self, benchmark):
        """Benchmark single neuron step performance."""
        neuron = LeakyIntegrateAndFire(0)
        
        def step_neuron():
            return neuron.step(1.0, 10.0)
        
        result = benchmark(step_neuron)
        
        # Store extra info
        benchmark.extra_info['neuron_type'] = 'LIF'
        benchmark.extra_info['throughput_neurons_per_sec'] = 1.0 / benchmark.stats['mean']
    
    def test_neuron_population_step(self, benchmark, network_size):
        """Benchmark neuron population step performance."""
        group = NeuronGroup(f"group_{network_size}", network_size, "lif")
        currents = np.random.uniform(5, 15, network_size)
        
        def step_population():
            return group.step(1.0, currents)
        
        result = benchmark(step_population)
        
        # Calculate throughput
        neurons_per_sec = network_size / benchmark.stats['mean']
        
        benchmark.extra_info['population_size'] = network_size
        benchmark.extra_info['throughput_neurons_per_sec'] = neurons_per_sec
    
    @pytest.mark.parametrize("dt", [0.1, 0.5, 1.0])
    def test_network_step_throughput(self, benchmark, small_network, dt):
        """Benchmark network step throughput with different time steps."""
        network = small_network
        total_neurons = sum(g.size for g in network.neuron_groups.values())
        
        def step_network():
            network.step(dt)
        
        result = benchmark(step_network)
        
        # Calculate throughput
        neurons_per_sec = total_neurons / benchmark.stats['mean']
        
        benchmark.extra_info['total_neurons'] = total_neurons
        benchmark.extra_info['dt_ms'] = dt
        benchmark.extra_info['throughput_neurons_per_sec'] = neurons_per_sec
    
    def test_scalability(self, benchmark):
        """Test throughput scalability across different network sizes."""
        sizes = [100, 500, 1000, 2000]
        throughputs = []
        
        def measure_throughput(size):
            network = Network(f"scale_{size}")
            network.add_neuron_group("layer", size, "lif")
            
            def step():
                network.step(1.0)
            
            # Quick measurement
            import time
            start = time.perf_counter()
            for _ in range(100):
                step()
            elapsed = time.perf_counter() - start
            
            return (size * 100) / elapsed
        
        result = benchmark(lambda: [measure_throughput(s) for s in sizes])
        
        benchmark.extra_info['network_sizes'] = sizes
        benchmark.extra_info['scalability_factor'] = max(result) / min(result) if result else 0


# -----------------------------------------------------------------------------
# Memory Footprint Benchmarks
# -----------------------------------------------------------------------------

class TestMemoryFootprint:
    """Benchmarks measuring memory usage."""
    
    def test_neuron_memory(self, benchmark, network_size, neuron_model):
        """Measure memory footprint of neuron populations."""
        process = psutil.Process()
        
        def create_neurons():
            gc.collect()
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            
            group = NeuronGroup(f"test_{network_size}", network_size, neuron_model)
            
            gc.collect()
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            
            return mem_after - mem_before, group
        
        mem_used, _ = benchmark(create_neurons)
        
        benchmark.extra_info['network_size'] = network_size
        benchmark.extra_info['neuron_model'] = neuron_model
        benchmark.extra_info['memory_mb'] = mem_used
        benchmark.extra_info['memory_per_neuron_kb'] = (mem_used * 1024) / network_size if network_size > 0 else 0
    
    def test_synapse_memory(self, benchmark):
        """Measure memory footprint of synaptic connections."""
        pre_size = 1000
        post_size = 1000
        connection_prob = 0.1
        
        process = psutil.Process()
        
        def create_synapses():
            gc.collect()
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            
            synapses = SynapseGroup(
                "test_synapses", pre_size, post_size, "stdp",
                connectivity="random", connection_probability=connection_prob
            )
            
            gc.collect()
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            
            num_connections = np.sum(synapses.connectivity)
            return mem_after - mem_before, num_connections
        
        mem_used, num_connections = benchmark(create_synapses)
        
        benchmark.extra_info['pre_size'] = pre_size
        benchmark.extra_info['post_size'] = post_size
        benchmark.extra_info['connection_probability'] = connection_prob
        benchmark.extra_info['num_connections'] = num_connections
        benchmark.extra_info['memory_mb'] = mem_used
        benchmark.extra_info['memory_per_synapse_bytes'] = (mem_used * 1024 * 1024) / num_connections if num_connections > 0 else 0
    
    def test_network_memory_scaling(self, benchmark):
        """Test memory scaling with network size."""
        sizes = [100, 500, 1000, 2000]
        process = psutil.Process()
        
        def measure_memory():
            memories = []
            
            for size in sizes:
                gc.collect()
                mem_before = process.memory_info().rss / 1024 / 1024
                
                network = Network(f"mem_test_{size}")
                layers = size // 100
                neurons_per_layer = size // max(layers, 1)
                
                for i in range(max(layers, 1)):
                    network.add_neuron_group(f"layer_{i}", neurons_per_layer, "lif")
                
                for i in range(max(layers - 1, 0)):
                    network.connect(f"layer_{i}", f"layer_{i+1}", connection_probability=0.1)
                
                gc.collect()
                mem_after = process.memory_info().rss / 1024 / 1024
                
                memories.append(mem_after - mem_before)
            
            return memories
        
        memories = benchmark(measure_memory)
        
        benchmark.extra_info['network_sizes'] = sizes
        benchmark.extra_info['memory_usage_mb'] = memories
        benchmark.extra_info['memory_scaling_factor'] = max(memories) / min(memories) if memories and min(memories) > 0 else 0


# -----------------------------------------------------------------------------
# Convergence Speed Benchmarks
# -----------------------------------------------------------------------------

class TestConvergenceSpeed:
    """Benchmarks measuring learning and convergence speed."""
    
    def test_pattern_learning_convergence(self, benchmark):
        """Test convergence speed for pattern learning task."""
        network = Network("pattern_learning")
        network.add_neuron_group("input", 100, "lif")
        network.add_neuron_group("output", 10, "lif")
        network.connect("input", "output", model="stdp", connection_probability=0.2)
        
        # Create target pattern
        target_pattern = np.random.choice([0, 1], size=10)
        
        def train_epoch():
            # Generate input pattern
            input_pattern = np.random.uniform(0, 20, 100)
            
            # Set input currents
            input_group = network.neuron_groups["input"]
            for i, current in enumerate(input_pattern):
                if i < len(input_group.neurons):
                    input_group.neurons[i].external_current = current
            
            # Run for 100ms
            for _ in range(100):
                network.step(1.0)
            
            # Check output spikes
            output_group = network.neuron_groups["output"]
            output_spikes = np.zeros(10)
            for i, neuron in enumerate(output_group.neurons[:10]):
                if hasattr(neuron, 'spike_count'):
                    output_spikes[i] = neuron.spike_count
            
            # Calculate error
            error = np.mean(np.abs(output_spikes > 0 - target_pattern))
            return error
        
        def train_until_convergence():
            errors = []
            max_epochs = 100
            
            for epoch in range(max_epochs):
                error = train_epoch()
                errors.append(error)
                
                # Check convergence (error < threshold)
                if error < 0.1:
                    return epoch + 1, errors
            
            return max_epochs, errors
        
        epochs_to_converge, error_history = benchmark(train_until_convergence)
        
        benchmark.extra_info['epochs_to_converge'] = epochs_to_converge
        benchmark.extra_info['final_error'] = error_history[-1] if error_history else 1.0
        benchmark.extra_info['convergence_rate'] = 1.0 / epochs_to_converge if epochs_to_converge > 0 else 0
    
    def test_sequence_learning_speed(self, benchmark):
        """Test learning speed for sequence recognition."""
        network = Network("sequence_learning")
        network.add_neuron_group("input", 50, "lif")
        network.add_neuron_group("hidden", 100, "adex")
        network.add_neuron_group("output", 10, "lif")
        
        network.connect("input", "hidden", model="stdp", connection_probability=0.15)
        network.connect("hidden", "output", model="stdp", connection_probability=0.1)
        
        # Create sequence
        sequence_length = 5
        sequence = [np.random.uniform(0, 20, 50) for _ in range(sequence_length)]
        
        def train_sequence():
            total_error = 0
            
            for step, pattern in enumerate(sequence):
                # Apply input
                input_group = network.neuron_groups["input"]
                for i, current in enumerate(pattern):
                    if i < len(input_group.neurons):
                        input_group.neurons[i].external_current = current
                
                # Step network
                for _ in range(20):  # 20ms per sequence item
                    network.step(1.0)
                
                # Measure activity
                output_group = network.neuron_groups["output"]
                output_activity = sum(1 for n in output_group.neurons if hasattr(n, 'is_spiking') and n.is_spiking)
                
                # Simple error: want different outputs for different sequence positions
                expected_active = (step + 1) * 2  # Different expectation per step
                total_error += abs(output_activity - expected_active)
            
            return total_error / sequence_length
        
        def measure_learning():
            errors = []
            for trial in range(50):
                error = train_sequence()
                errors.append(error)
            
            # Calculate learning rate (reduction in error)
            if len(errors) > 1:
                learning_rate = (errors[0] - errors[-1]) / errors[0] if errors[0] > 0 else 0
            else:
                learning_rate = 0
            
            return learning_rate, errors
        
        learning_rate, error_history = benchmark(measure_learning)
        
        benchmark.extra_info['learning_rate'] = learning_rate
        benchmark.extra_info['initial_error'] = error_history[0] if error_history else 0
        benchmark.extra_info['final_error'] = error_history[-1] if error_history else 0
        benchmark.extra_info['sequence_length'] = sequence_length
    
    def test_homeostatic_adaptation_speed(self, benchmark):
        """Test speed of homeostatic adaptation."""
        network = Network("homeostatic")
        network.add_neuron_group("adaptive", 100, "adex")
        
        target_rate = 10.0  # Hz
        
        def adapt_network():
            adaptation_steps = []
            
            for step in range(100):
                # Variable input
                input_strength = 5.0 + step * 0.1
                currents = np.random.uniform(0, input_strength, 100)
                
                # Run for 100ms
                spike_count = 0
                group = network.neuron_groups["adaptive"]
                
                for _ in range(100):
                    spikes = group.step(1.0, currents)
                    spike_count += np.sum(spikes)
                
                # Calculate firing rate
                firing_rate = spike_count / (0.1 * 100)  # Hz
                
                # Measure distance from target
                error = abs(firing_rate - target_rate)
                adaptation_steps.append(error)
                
                # Simple adaptation: adjust excitability
                if firing_rate < target_rate:
                    for neuron in group.neurons:
                        if hasattr(neuron, 'v_thresh'):
                            neuron.v_thresh *= 0.99  # Lower threshold
                else:
                    for neuron in group.neurons:
                        if hasattr(neuron, 'v_thresh'):
                            neuron.v_thresh *= 1.01  # Raise threshold
                
                # Check if adapted
                if error < 1.0:
                    return step + 1, adaptation_steps
            
            return 100, adaptation_steps
        
        steps_to_adapt, error_history = benchmark(adapt_network)
        
        benchmark.extra_info['steps_to_adapt'] = steps_to_adapt
        benchmark.extra_info['target_rate_hz'] = target_rate
        benchmark.extra_info['final_error_hz'] = error_history[-1] if error_history else 0
        benchmark.extra_info['adaptation_speed'] = 1.0 / steps_to_adapt if steps_to_adapt > 0 else 0


# -----------------------------------------------------------------------------
# Integration and System Benchmarks
# -----------------------------------------------------------------------------

class TestSystemPerformance:
    """Comprehensive system-level benchmarks."""
    
    def test_full_simulation_benchmark(self, benchmark, large_network):
        """Benchmark full simulation with recording."""
        simulator = Simulator(large_network, mode=SimulationMode.TIME_STEP, dt=1.0)
        
        def run_simulation():
            return simulator.run(duration=100.0, record=True)
        
        results = benchmark(run_simulation)
        
        if 'performance' in results:
            perf = results['performance']
            benchmark.extra_info['simulation_time_ms'] = perf.get('simulation_time', 0)
            benchmark.extra_info['real_time_s'] = perf.get('real_time', 0)
            benchmark.extra_info['time_factor'] = perf['simulation_time'] / (perf['real_time'] * 1000) if perf.get('real_time', 0) > 0 else 0
        
        total_neurons = sum(g.size for g in large_network.neuron_groups.values())
        benchmark.extra_info['total_neurons'] = total_neurons
    
    @pytest.mark.parametrize("mode", [SimulationMode.TIME_STEP, SimulationMode.EVENT_DRIVEN])
    def test_simulation_modes(self, benchmark, small_network, mode):
        """Compare performance of different simulation modes."""
        simulator = Simulator(small_network, mode=mode, dt=0.1)
        
        def run_simulation():
            return simulator.run(duration=50.0, record=False)
        
        results = benchmark(run_simulation)
        
        benchmark.extra_info['simulation_mode'] = mode.value
        if 'performance' in results:
            benchmark.extra_info['total_steps'] = results['performance'].get('total_steps', 0)
            benchmark.extra_info['total_events'] = results['performance'].get('total_events', 0)
    
    def test_parallel_network_simulation(self, benchmark):
        """Test performance with multiple independent networks."""
        networks = []
        for i in range(4):
            net = Network(f"parallel_{i}")
            net.add_neuron_group("layer", 250, "lif")
            networks.append(net)
        
        def simulate_all():
            results = []
            for net in networks:
                sim = Simulator(net, dt=1.0)
                results.append(sim.run(duration=50.0, record=False))
            return results
        
        all_results = benchmark(simulate_all)
        
        benchmark.extra_info['num_networks'] = len(networks)
        benchmark.extra_info['neurons_per_network'] = 250
        benchmark.extra_info['total_neurons'] = 250 * len(networks)


# -----------------------------------------------------------------------------
# Benchmark Configuration
# -----------------------------------------------------------------------------

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "memory: marks tests that measure memory usage"
    )
    config.addinivalue_line(
        "markers", "convergence: marks tests that measure convergence speed"
    )


if __name__ == "__main__":
    # Run with: pytest pytest_benchmarks.py --benchmark-only
    pytest.main([__file__, "--benchmark-only", "-v"])
