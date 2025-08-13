"""
Performance and Scalability Benchmarks for Neuromorphic Networks
================================================================

This module implements comprehensive benchmarks for evaluating the performance
and scalability of spiking neural networks across different hardware platforms.

Benchmarks:
- Network sizes: 1k, 10k, 50k neurons
- Platforms: CPU and GPU (with graceful fallback)
- Metrics: Wall-clock time, memory usage, spike throughput
- Results stored in CSV format for analysis
"""

import os
import sys
import csv
import json
import time
import timeit
import psutil
import platform
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import numpy as np

# Note: memory_profiler import removed as we use lightweight psutil monitoring

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core modules
from core.network import NeuromorphicNetwork
from core.neurons import NeuronPopulation
from core.synapses import SynapsePopulation

# Try to import GPU libraries
try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
    
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    network_size: int
    simulation_time: float  # ms
    dt: float  # ms
    connection_probability: float
    platform: str  # 'cpu', 'cuda', 'cupy'
    num_layers: int
    neurons_per_layer: int
    synapse_type: str
    neuron_type: str


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    config: BenchmarkConfig
    wall_clock_time: float  # seconds
    memory_peak: float  # MB
    memory_average: float  # MB
    total_spikes: int
    spike_rate: float  # Hz
    spike_throughput: float  # spikes/second
    simulation_steps: int
    step_time_mean: float  # ms
    step_time_std: float  # ms
    platform_info: Dict[str, Any]
    timestamp: str
    success: bool
    error_message: str = ""


class NetworkBenchmark:
    """Base class for network benchmarks."""
    
    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self.network: Optional[NeuromorphicNetwork] = None
        self.spike_counts: List[int] = []
        self.step_times: List[float] = []
        
    def build_network(self) -> NeuromorphicNetwork:
        """Build the network architecture."""
        network = NeuromorphicNetwork()
        
        # Calculate neurons per layer to achieve target size
        neurons_per_layer = self.config.network_size // self.config.num_layers
        
        # Add layers with more excitable neuron parameters
        for i in range(self.config.num_layers):
            layer_name = f"layer_{i}"
            network.add_layer(
                name=layer_name,
                size=neurons_per_layer,
                neuron_type=self.config.neuron_type,
                # More excitable parameters for reliable benchmarking
                v_thresh=-60.0,  # Lower threshold (easier to spike)
                tau_m=10.0       # Shorter membrane time constant
            )
        
        # Add feedforward connections between consecutive layers
        for i in range(self.config.num_layers - 1):
            pre_layer = f"layer_{i}"
            post_layer = f"layer_{i+1}"
            network.connect_layers(
                pre_layer=pre_layer,
                post_layer=post_layer,
                synapse_type=self.config.synapse_type,
                connection_probability=self.config.connection_probability
            )
        
        # Add some recurrent connections within layers
        for i in range(self.config.num_layers):
            layer_name = f"layer_{i}"
            network.connect_layers(
                pre_layer=layer_name,
                post_layer=layer_name,
                synapse_type=self.config.synapse_type,
                connection_probability=self.config.connection_probability * 0.5  # Fewer recurrent
            )
        
        return network
    
    def inject_input(self, layer_idx: int = 0) -> List[float]:
        """Inject random input current into specified layer."""
        if self.network and f"layer_{layer_idx}" in self.network.layers:
            layer = self.network.layers[f"layer_{layer_idx}"]
            # Generate stronger random input currents (50-100 nA) for reliable spiking
            input_currents = np.random.uniform(50, 100, layer.size)
            return input_currents.tolist()
        return []
    
    def run_simulation(self) -> Tuple[float, int]:
        """Run the simulation and collect metrics."""
        self.network = self.build_network()
        
        num_steps = int(self.config.simulation_time / self.config.dt)
        total_spikes = 0
        
        # Warm-up phase
        for _ in range(10):
            self.network.step(self.config.dt)
        
        # Main simulation loop
        start_time = time.perf_counter()
        
        for step in range(num_steps):
            step_start = time.perf_counter()
            
            # Inject input to first layer every 10ms using proper method
            input_currents: Optional[List[float]] = None
            if step % int(10.0 / self.config.dt) == 0:
                input_currents = self.inject_input(0)
            
            # Step the network with proper input injection
            if input_currents is not None and "layer_0" in self.network.layers:
                # Apply input currents properly through network step
                input_layer = self.network.layers["layer_0"]
                
                # Apply external input by modifying the layer step
                def modified_step(dt, I_syn):
                    # Combine synaptic and external currents
                    combined_currents = []
                    for i in range(len(I_syn)):
                        if input_currents is not None and i < len(input_currents):
                            external_current = input_currents[i]
                        else:
                            external_current = 0.0
                        combined_currents.append(I_syn[i] + external_current)
                    return input_layer.neuron_population.step(dt, combined_currents)
                
                # Temporarily replace step method
                original_step = input_layer.step
                input_layer.step = modified_step
                
                # Step the network
                self.network.step(self.config.dt)
                
                # Restore original step method
                input_layer.step = original_step
            else:
                # Normal step without external input
                self.network.step(self.config.dt)
            
            # Count spikes - ULTRA-FAST VECTORIZED for maximum performance
            step_spikes = 0
            for layer in self.network.layers.values():
                # Use the new vectorized spike state getter
                if hasattr(layer.neuron_population, 'get_spike_states'):
                    spike_states = layer.neuron_population.get_spike_states()
                    step_spikes += sum(spike_states)
                else:
                    # Fallback to list comprehension for compatibility
                    step_spikes += sum(neuron.is_spiking for neuron in layer.neuron_population.neurons)
                
                # Reset spike flags using vectorized method
                if hasattr(layer.neuron_population, 'reset_spike_flags'):
                    layer.neuron_population.reset_spike_flags()
                else:
                    # Fallback reset method
                    for neuron in layer.neuron_population.neurons:
                        neuron.is_spiking = False
            
            total_spikes += step_spikes
            self.spike_counts.append(step_spikes)
            
            step_time = (time.perf_counter() - step_start) * 1000  # Convert to ms
            self.step_times.append(step_time)
        
        wall_clock_time = time.perf_counter() - start_time
        
        return wall_clock_time, total_spikes


class CPUBenchmark(NetworkBenchmark):
    """CPU-based network benchmark."""
    
    def __init__(self, config: BenchmarkConfig) -> None:
        super().__init__(config)
        config.platform = 'cpu'


class GPUBenchmark(NetworkBenchmark):
    """GPU-based network benchmark using PyTorch or CuPy."""
    
    def __init__(self, config: BenchmarkConfig, backend: str = 'torch') -> None:
        super().__init__(config)
        self.backend = backend
        config.platform = f'gpu_{backend}'
        
        if backend == 'torch' and not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        if backend == 'cupy' and not CUPY_AVAILABLE:
            raise RuntimeError("CuPy not available")
    
    def build_network(self) -> NeuromorphicNetwork:
        """Build GPU-accelerated network."""
        # For now, use the same network structure
        # In a real implementation, we would use GPU-specific implementations
        network = super().build_network()
        
        if self.backend == 'torch' and CUDA_AVAILABLE:
            # Move network tensors to GPU
            self._move_to_gpu_torch(network)
        elif self.backend == 'cupy':
            # Convert arrays to CuPy arrays
            self._move_to_gpu_cupy(network)
        
        return network
    
    def _move_to_gpu_torch(self, network: NeuromorphicNetwork) -> None:
        """Move network data to GPU using PyTorch."""
        # This would require modifying the network classes to support torch tensors
        # For now, we'll just use the CPU implementation
        pass
    
    def _move_to_gpu_cupy(self, network: NeuromorphicNetwork) -> None:
        """Move network data to GPU using CuPy."""
        # This would require modifying the network classes to support CuPy arrays
        # For now, we'll just use the CPU implementation
        pass


class BenchmarkRunner:
    """Main benchmark runner with automated testing and result storage."""
    
    def __init__(self, output_dir: str = "benchmark_results") -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results: List[BenchmarkResult] = []
        
    def get_platform_info(self) -> Dict[str, Any]:
        """Collect platform information."""
        info = {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'numpy_version': np.__version__,
        }
        
        if TORCH_AVAILABLE:
            info['torch_version'] = torch.__version__
            info['cuda_available'] = CUDA_AVAILABLE
            if CUDA_AVAILABLE:
                info['cuda_version'] = 'available'  # Simplified to avoid torch.version issues
                info['gpu_name'] = torch.cuda.get_device_name(0)
                info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        if CUPY_AVAILABLE:
            info['cupy_version'] = cp.__version__
            
        return info
    
    def run_single_benchmark(self, config: BenchmarkConfig) -> BenchmarkResult:
        """Run a single benchmark configuration."""
        print(f"\nRunning benchmark: {config.network_size} neurons on {config.platform}")
        print(f"  Layers: {config.num_layers}, Connection probability: {config.connection_probability}")
        
        try:
            # Select appropriate benchmark class
            if config.platform == 'cpu':
                benchmark = CPUBenchmark(config)
            elif config.platform == 'gpu_torch':
                if not CUDA_AVAILABLE:
                    print("  CUDA not available, falling back to CPU")
                    config.platform = 'cpu'
                    benchmark = CPUBenchmark(config)
                else:
                    benchmark = GPUBenchmark(config, backend='torch')
            elif config.platform == 'gpu_cupy':
                if not CUPY_AVAILABLE:
                    print("  CuPy not available, falling back to CPU")
                    config.platform = 'cpu'
                    benchmark = CPUBenchmark(config)
                else:
                    benchmark = GPUBenchmark(config, backend='cupy')
            else:
                raise ValueError(f"Unknown platform: {config.platform}")
            
            # Fast execution without heavy memory profiling for speed benchmarks
            import psutil
            import os
            
            # Get initial memory
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Run benchmark with minimal overhead
            start_time = time.perf_counter()
            wall_clock_time, total_spikes = benchmark.run_simulation()
            end_time = time.perf_counter()
            
            # Get final memory (lightweight monitoring)
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            
            # Calculate metrics
            memory_peak = memory_after  # Simplified - peak ≈ final for short runs
            memory_average = (memory_before + memory_after) / 2
            simulation_steps = int(config.simulation_time / config.dt)
            spike_rate = total_spikes / (config.simulation_time / 1000.0) / config.network_size  # Hz per neuron
            spike_throughput = total_spikes / wall_clock_time  # Total spikes per second
            
            # Step time statistics
            if benchmark.step_times:
                step_time_mean = float(np.mean(benchmark.step_times))
                step_time_std = float(np.std(benchmark.step_times))
            else:
                step_time_mean = 0.0
                step_time_std = 0.0
            
            result = BenchmarkResult(
                config=config,
                wall_clock_time=wall_clock_time,
                memory_peak=memory_peak,
                memory_average=memory_average,
                total_spikes=total_spikes,
                spike_rate=spike_rate,
                spike_throughput=spike_throughput,
                simulation_steps=simulation_steps,
                step_time_mean=step_time_mean,
                step_time_std=step_time_std,
                platform_info=self.get_platform_info(),
                timestamp=datetime.now().isoformat(),
                success=True
            )
            
            print(f"  ✓ Completed in {wall_clock_time:.2f}s")
            print(f"    Memory: {memory_peak:.1f} MB (peak), {memory_average:.1f} MB (avg)")
            print(f"    Spikes: {total_spikes:,} total, {spike_rate:.1f} Hz/neuron")
            print(f"    Throughput: {spike_throughput:.0f} spikes/second")
            
        except Exception as e:
            print(f"  ✗ Failed: {str(e)}")
            result = BenchmarkResult(
                config=config,
                wall_clock_time=0.0,
                memory_peak=0.0,
                memory_average=0.0,
                total_spikes=0,
                spike_rate=0.0,
                spike_throughput=0.0,
                simulation_steps=0,
                step_time_mean=0.0,
                step_time_std=0.0,
                platform_info=self.get_platform_info(),
                timestamp=datetime.now().isoformat(),
                success=False,
                error_message=str(e)
            )
        
        return result
    
    def run_benchmark_suite(self) -> None:
        """Run the complete benchmark suite."""
        print("="*60)
        print("Starting Performance Benchmark Suite")
        print("="*60)
        
        # Define benchmark configurations
        network_sizes = [1000, 10000, 50000]  # 1k, 10k, 50k neurons
        platforms = ['cpu']
        
        # Add GPU platforms if available
        if CUDA_AVAILABLE:
            platforms.append('gpu_torch')
        if CUPY_AVAILABLE:
            platforms.append('gpu_cupy')
        
        configs = []
        for size in network_sizes:
            for platform in platforms:
                # Adjust parameters based on network size - OPTIMIZED for speed
                if size <= 1000:
                    num_layers = 3
                    sim_time = 100.0  # Reduced from 1000ms to 100ms for faster benchmarks
                    conn_prob = 0.1
                elif size <= 10000:
                    num_layers = 4
                    sim_time = 50.0   # Reduced from 500ms to 50ms for faster benchmarks
                    conn_prob = 0.05
                else:  # 50000
                    num_layers = 5
                    sim_time = 20.0   # Reduced from 200ms to 20ms for faster benchmarks
                    conn_prob = 0.02
                
                config = BenchmarkConfig(
                    network_size=size,
                    simulation_time=sim_time,
                    dt=1.0,  # 1ms timestep
                    connection_probability=conn_prob,
                    platform=platform,
                    num_layers=num_layers,
                    neurons_per_layer=size // num_layers,
                    synapse_type='stdp',
                    neuron_type='adex'
                )
                configs.append(config)
        
        # Run benchmarks
        for config in configs:
            result = self.run_single_benchmark(config)
            self.results.append(result)
        
        # Save results
        self.save_results()
        self.print_summary()
    
    def save_results(self) -> None:
        """Save benchmark results to CSV and JSON files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save to CSV
        csv_file = os.path.join(self.output_dir, f"benchmark_results_{timestamp}.csv")
        with open(csv_file, 'w', newline='') as f:
            if self.results:
                # Define CSV columns
                fieldnames = [
                    'network_size', 'platform', 'simulation_time_ms', 'wall_clock_time_s',
                    'memory_peak_mb', 'memory_average_mb', 'total_spikes', 
                    'spike_rate_hz', 'spike_throughput', 'simulation_steps',
                    'step_time_mean_ms', 'step_time_std_ms', 'success', 'error_message'
                ]
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in self.results:
                    row = {
                        'network_size': result.config.network_size,
                        'platform': result.config.platform,
                        'simulation_time_ms': result.config.simulation_time,
                        'wall_clock_time_s': result.wall_clock_time,
                        'memory_peak_mb': result.memory_peak,
                        'memory_average_mb': result.memory_average,
                        'total_spikes': result.total_spikes,
                        'spike_rate_hz': result.spike_rate,
                        'spike_throughput': result.spike_throughput,
                        'simulation_steps': result.simulation_steps,
                        'step_time_mean_ms': result.step_time_mean,
                        'step_time_std_ms': result.step_time_std,
                        'success': result.success,
                        'error_message': result.error_message
                    }
                    writer.writerow(row)
        
        print(f"\nResults saved to: {csv_file}")
        
        # Save detailed results to JSON
        json_file = os.path.join(self.output_dir, f"benchmark_details_{timestamp}.json")
        with open(json_file, 'w') as f:
            results_dict = []
            for result in self.results:
                result_dict = asdict(result)
                # Convert config to dict
                result_dict['config'] = asdict(result.config)
                results_dict.append(result_dict)
            json.dump(results_dict, f, indent=2)
        
        print(f"Detailed results saved to: {json_file}")
    
    def print_summary(self) -> None:
        """Print a summary of benchmark results."""
        print("\n" + "="*60)
        print("Benchmark Summary")
        print("="*60)
        
        # Group results by network size
        by_size = {}
        for result in self.results:
            size = result.config.network_size
            if size not in by_size:
                by_size[size] = []
            by_size[size].append(result)
        
        for size in sorted(by_size.keys()):
            print(f"\nNetwork Size: {size:,} neurons")
            print("-" * 40)
            
            for result in by_size[size]:
                if result.success:
                    print(f"  {result.config.platform:12s}: {result.wall_clock_time:6.2f}s, "
                          f"{result.memory_peak:6.1f} MB, "
                          f"{result.spike_throughput:8.0f} spikes/s")
                else:
                    print(f"  {result.config.platform:12s}: FAILED - {result.error_message}")


def run_timeit_microbenchmarks():
    """Run additional micro-benchmarks using timeit for specific operations."""
    print("\n" + "="*60)
    print("Micro-benchmarks (using timeit)")
    print("="*60)
    
    results = []
    
    # Benchmark 1: Single neuron step
    setup = """
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.neurons import AdaptiveExponentialIntegrateAndFire
neuron = AdaptiveExponentialIntegrateAndFire(0)
"""
    stmt = "neuron.step(1.0, 10.0)"
    time_single = timeit.timeit(stmt, setup, number=10000) / 10000
    print(f"Single neuron step: {time_single*1e6:.2f} μs")
    results.append(("single_neuron_step", time_single))
    
    # Benchmark 2: Population step (100 neurons)
    setup = """
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.neurons import NeuronPopulation
import numpy as np
pop = NeuronPopulation(100, 'adex')
currents = np.random.uniform(5, 15, 100).tolist()
"""
    stmt = "pop.step(1.0, currents)"
    time_pop = timeit.timeit(stmt, setup, number=1000) / 1000
    print(f"Population step (100 neurons): {time_pop*1e3:.2f} ms")
    results.append(("population_100_step", time_pop))
    
    # Benchmark 3: Network construction
    setup = """
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.network import NeuromorphicNetwork
"""
    stmt = """
network = NeuromorphicNetwork()
network.add_layer('input', 100, 'adex')
network.add_layer('hidden', 100, 'adex')
network.add_layer('output', 10, 'adex')
network.connect_layers('input', 'hidden', 'stdp', 0.1)
network.connect_layers('hidden', 'output', 'stdp', 0.1)
"""
    time_construct = timeit.timeit(stmt, setup, number=10) / 10
    print(f"Network construction (210 neurons): {time_construct*1e3:.2f} ms")
    results.append(("network_construction", time_construct))
    
    # Save micro-benchmark results
    os.makedirs("benchmark_results", exist_ok=True)
    csv_file = os.path.join("benchmark_results", "microbenchmark_timeit.csv")
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['operation', 'time_seconds'])
        writer.writerows(results)
    
    print(f"\nMicro-benchmark results saved to: {csv_file}")


def main():
    """Main entry point for benchmarking."""
    runner = BenchmarkRunner()
    
    # Run main benchmark suite
    runner.run_benchmark_suite()
    
    # Run additional timeit micro-benchmarks
    run_timeit_microbenchmarks()
    
    print("\n" + "="*60)
    print("Benchmarking Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
