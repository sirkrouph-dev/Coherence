"""
#!/usr/bin/env python3
"""
GPU-Accelerated Neuron Models with Enhanced Scaling
====================================================

This module provides GPU-optimized neuron implementations that can scale
to millions of neurons while maintaining biological plausibility.
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import cupy as cp
    import cupyx
    from cupy.cuda import MemoryPool, memory

    GPU_AVAILABLE = True
    print("CuPy GPU acceleration available")
except ImportError:
    cp = np  # Fallback to NumPy
    GPU_AVAILABLE = False
    print("[WARNING] CuPy not available, using CPU fallback")

try:
    import torch
    import torch.nn as nn
    import torch.cuda as cuda

    TORCH_GPU_AVAILABLE = torch.cuda.is_available()
    if TORCH_GPU_AVAILABLE:
        print(
            f"PyTorch GPU acceleration available ({torch.cuda.get_device_name(0)})"
        )
except ImportError:
    TORCH_GPU_AVAILABLE = False
    print("[WARNING] PyTorch not available")


@dataclass
class GPUMetrics:
    """Container for GPU performance metrics."""

    gpu_memory_used: float  # MB
    gpu_memory_total: float  # MB
    gpu_utilization: float  # %
    compute_time: float  # seconds
    neurons_processed: int
    spikes_generated: int
    throughput: float  # neurons/second


class GPUNeuronPool:
    """
    Manages large pools of neurons on GPU with efficient batch processing.
    """

    def __init__(
        self,
        num_neurons: int,
        neuron_type: str = "adex",
        use_gpu: bool = True,
        batch_size: Optional[int] = None,  # Auto-determine optimal batch size
        precision: str = "float32",
        max_spike_history: int = 10000,  # Add memory limit
    ):
        """
        Initialize GPU neuron pool.

        Args:
            num_neurons: Number of neurons to simulate
            neuron_type: Type of neuron model ('adex', 'lif', 'izhikevich')
            use_gpu: Whether to use GPU acceleration
            batch_size: Number of neurons to process in parallel
            precision: Numerical precision ('float16', 'float32', 'float64')
            max_spike_history: Maximum number of spike events to store (prevents memory leak)
        """
        self.num_neurons = num_neurons
        self.neuron_type = neuron_type
        self.use_gpu = use_gpu and GPU_AVAILABLE
        
        # Determine optimal batch size for massive scale
        if batch_size is None:
            if self.use_gpu:
                # For GPU: process all neurons at once for better performance
                # Only batch if > 10M neurons to avoid memory issues
                if num_neurons <= 10_000_000:
                    self.batch_size = num_neurons  # Process all at once!
                else:
                    self.batch_size = 1_000_000  # 1M batch for massive networks
            else:
                self.batch_size = min(50_000, num_neurons)  # CPU conservative
        else:
            self.batch_size = min(batch_size, num_neurons)
        self.precision = precision
        self.max_spike_history = max_spike_history

        # Set up compute backend
        self.xp = cp if self.use_gpu else np
        self.device = "gpu" if self.use_gpu else "cpu"

        # Set dtype based on precision
        self.dtype = {
            "float16": self.xp.float16,
            "float32": self.xp.float32,
            "float64": self.xp.float64,
        }.get(precision, self.xp.float32)

        # Initialize neuron states
        self._initialize_neurons()

        # Performance tracking
        self.metrics = []
        self.total_spikes = 0

        print(f"Initialized {num_neurons:,} {neuron_type} neurons on {self.device}")
        print(f"  Batch size: {self.batch_size:,}")
        print(f"  Precision: {precision}")

        if self.use_gpu:
            self._print_gpu_info()

    def _print_gpu_info(self):
        """Print GPU memory and device information."""
        if self.use_gpu:
            mempool = cp.get_default_memory_pool()
            used_bytes = mempool.used_bytes()
            total_bytes = mempool.total_bytes()

            # Get device properties
            device = cp.cuda.Device()
            props = device.attributes

            print(f"GPU Device Information:")
            print(f"  Name: {props.get('DeviceName', 'Unknown')}")
            print(f"  Memory Used: {used_bytes / 1024**2:.2f} MB")
            print(f"  Memory Allocated: {total_bytes / 1024**2:.2f} MB")
            print(f"  Compute Capability: {device.compute_capability}")

    def _initialize_neurons(self):
        """Initialize neuron state variables."""
        n = self.num_neurons

        if self.neuron_type == "adex":
            # Adaptive Exponential Integrate-and-Fire parameters
            self.v = self.xp.full(n, -65.0, dtype=self.dtype)  # Membrane potential
            self.w = self.xp.zeros(n, dtype=self.dtype)  # Adaptation variable
            self.v_rest = -65.0
            self.v_thresh = -55.0
            self.v_reset = -65.0
            self.tau_m = 20.0
            self.tau_w = 144.0
            self.delta_t = 2.0
            self.a = 4.0
            self.b = 0.0805

        elif self.neuron_type == "lif":
            # Leaky Integrate-and-Fire parameters
            self.v = self.xp.full(n, -65.0, dtype=self.dtype)
            self.v_rest = -65.0
            self.v_thresh = -55.0
            self.v_reset = -70.0
            self.tau_m = 10.0

        elif self.neuron_type == "izhikevich":
            # Izhikevich neuron parameters
            self.v = self.xp.full(n, -65.0, dtype=self.dtype)
            self.u = self.xp.full(n, -14.0, dtype=self.dtype)
            self.a = 0.02
            self.b = 0.2
            self.c = -65.0
            self.d = 8.0

        # Common state variables with circular buffer for memory management
        self.spike_times = []  # Will use circular buffer logic
        self.spike_indices = []  # Will use circular buffer logic
        self.spike_history_index = 0  # Current position in circular buffer
        self.refractory = self.xp.zeros(n, dtype=self.xp.bool_)
        self.refractory_time = self.xp.zeros(n, dtype=self.dtype)

    def step(
        self, dt: float, I_syn: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Advance all neurons by one time step.

        Args:
            dt: Time step in milliseconds
            I_syn: Synaptic current for each neuron (optional)

        Returns:
            Tuple of (spike_indices, metrics_dict)
        """
        start_time = time.time()

        # Initialize synaptic current if not provided
        if I_syn is None:
            I_syn = self.xp.zeros(self.num_neurons, dtype=self.dtype)
        elif not isinstance(I_syn, self.xp.ndarray):
            I_syn = self.xp.asarray(I_syn, dtype=self.dtype)

        # Update refractory periods
        self.refractory_time -= dt
        self.refractory = self.refractory_time > 0

        # Neuron-specific dynamics
        if self.neuron_type == "adex":
            spikes = self._step_adex(dt, I_syn)
        elif self.neuron_type == "lif":
            spikes = self._step_lif(dt, I_syn)
        elif self.neuron_type == "izhikevich":
            spikes = self._step_izhikevich(dt, I_syn)
        else:
            raise ValueError(f"Unknown neuron type: {self.neuron_type}")

        # Record spikes with circular buffer to prevent memory leak
        spike_indices = self.xp.where(spikes)[0]
        if len(spike_indices) > 0:
            # Implement circular buffer for spike history
            if len(self.spike_indices) >= self.max_spike_history:
                # Overwrite oldest entry in circular fashion
                self.spike_indices[
                    self.spike_history_index % self.max_spike_history
                ] = spike_indices
                self.spike_times[self.spike_history_index % self.max_spike_history] = (
                    self.xp.full(len(spike_indices), time.time())
                )
            else:
                # Still filling up the buffer
                self.spike_indices.append(spike_indices)
                self.spike_times.append(self.xp.full(len(spike_indices), time.time()))

            self.spike_history_index += 1
            self.total_spikes += len(spike_indices)

            # Periodic memory cleanup on GPU
            if self.use_gpu and self.spike_history_index % 1000 == 0:
                self._compact_memory()

        # Calculate metrics
        compute_time = time.time() - start_time
        metrics = self._calculate_metrics(compute_time, len(spike_indices))

        return spike_indices, metrics

    def _step_adex(self, dt: float, I_syn: np.ndarray) -> np.ndarray:
        """Step adaptive exponential integrate-and-fire neurons."""
        # Only update non-refractory neurons
        active = ~self.refractory

        if self.xp.any(active):
            # Exponential term with numerical stability
            exp_term = self.xp.exp(
                self.xp.clip(
                    (self.v[active] - self.v_thresh) / self.delta_t,
                    -10,
                    10,  # Prevent overflow
                )
            )

            # Update membrane potential
            dv = (
                -(self.v[active] - self.v_rest)
                + self.delta_t * exp_term
                - self.w[active]
                + I_syn[active]
            ) / self.tau_m
            self.v[active] += dv * dt

            # Update adaptation current
            dw = (self.a * (self.v[active] - self.v_rest) - self.w[active]) / self.tau_w
            self.w[active] += dw * dt

        # Check for spikes
        spikes = self.v >= self.v_thresh

        # Handle spikes
        if self.xp.any(spikes):
            self.v[spikes] = self.v_reset
            self.w[spikes] += self.b
            self.refractory_time[spikes] = 2.0  # 2ms refractory period

        return spikes

    def _step_lif(self, dt: float, I_syn: np.ndarray) -> np.ndarray:
        """Step leaky integrate-and-fire neurons - Optimized GPU version."""
        # Vectorized computation for all neurons at once
        active_mask = ~self.refractory
        
        # Update membrane potential for all neurons (vectorized)
        leak = -(self.v - self.v_rest) / self.tau_m
        self.v += dt * (leak + I_syn) * active_mask
        
        # Check for spikes (vectorized)
        spikes = self.v >= self.v_thresh
        
        # Handle spikes (vectorized)
        self.v = self.xp.where(spikes, self.v_reset, self.v)
        self.refractory_time = self.xp.where(spikes, 2.0, self.refractory_time)
        
        return spikes

    def _step_izhikevich(self, dt: float, I_syn: np.ndarray) -> np.ndarray:
        """Step Izhikevich neurons."""
        active = ~self.refractory

        if self.xp.any(active):
            # Update membrane potential and recovery variable
            v = self.v[active]
            u = self.u[active]

            dv = 0.04 * v * v + 5 * v + 140 - u + I_syn[active]
            du = self.a * (self.b * v - u)

            self.v[active] += dv * dt
            self.u[active] += du * dt

        # Check for spikes
        spikes = self.v >= 30.0  # Izhikevich spike threshold

        # Handle spikes
        if self.xp.any(spikes):
            self.v[spikes] = self.c
            self.u[spikes] += self.d
            self.refractory_time[spikes] = 1.0

        return spikes

    def _compact_memory(self):
        """Compact GPU memory periodically to prevent fragmentation."""
        if self.use_gpu:
            try:
                mempool = cp.get_default_memory_pool()
                # Only free unused blocks, not all blocks
                mempool.free_all_free_blocks()
            except Exception as e:
                print(f"Warning: Memory compaction failed: {e}")

    def _calculate_metrics(self, compute_time: float, num_spikes: int) -> Dict:
        """Calculate performance metrics."""
        metrics = {
            "compute_time": compute_time,
            "neurons_processed": self.num_neurons,
            "spikes_generated": num_spikes,
            "throughput": self.num_neurons / compute_time if compute_time > 0 else 0,
            "spike_rate": num_spikes / self.num_neurons if self.num_neurons > 0 else 0,
        }

        if self.use_gpu:
            mempool = cp.get_default_memory_pool()
            metrics["gpu_memory_used_mb"] = mempool.used_bytes() / 1024**2
            metrics["gpu_memory_total_mb"] = mempool.total_bytes() / 1024**2

            # Try to get GPU utilization (requires pynvml)
            try:
                import pynvml

                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                metrics["gpu_utilization"] = util.gpu
                metrics["gpu_memory_utilization"] = util.memory
            except Exception as e:
                # More specific error handling
                metrics["gpu_utilization"] = -1
                metrics["gpu_memory_utilization"] = -1
                metrics["gpu_metrics_error"] = str(e)

        self.metrics.append(metrics)
        return metrics

    def get_spike_statistics(self) -> Dict:
        """Get comprehensive spike statistics."""
        if len(self.spike_indices) == 0:
            return {
                "total_spikes": 0,
                "mean_spike_rate": 0,
                "std_spike_rate": 0,
                "active_neurons": 0,
                "max_spike_count": 0,
                "min_spike_count": 0,
                "silent_neurons": self.num_neurons,
            }

        # Concatenate all spike indices
        all_spikes = (
            self.xp.concatenate(self.spike_indices)
            if self.use_gpu
            else np.concatenate(self.spike_indices)
        )

        # Calculate statistics
        unique_neurons = self.xp.unique(all_spikes)
        spike_counts = self.xp.bincount(all_spikes, minlength=self.num_neurons)

        stats = {
            "total_spikes": self.total_spikes,
            "mean_spike_rate": float(self.xp.mean(spike_counts)),
            "std_spike_rate": float(self.xp.std(spike_counts)),
            "active_neurons": len(unique_neurons),
            "max_spike_count": int(self.xp.max(spike_counts)),
            "min_spike_count": (
                int(self.xp.min(spike_counts[spike_counts > 0]))
                if self.xp.any(spike_counts > 0)
                else 0
            ),
            "silent_neurons": int(self.xp.sum(spike_counts == 0)),
        }

        # Convert to CPU if needed
        if self.use_gpu:
            for key in stats:
                if isinstance(stats[key], cp.ndarray):
                    stats[key] = stats[key].get()

        return stats

    def clear_gpu_memory(self):
        """Clear GPU memory pools."""
        if self.use_gpu:
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
            print("GPU memory cleared")

    def to_cpu(self) -> Dict:
        """Transfer all data to CPU."""
        if self.use_gpu:
            return {
                "membrane_potential": self.v.get(),
                "spike_indices": [s.get() for s in self.spike_indices],
                "spike_times": [t.get() for t in self.spike_times],
                "total_spikes": self.total_spikes,
                "metrics": self.metrics,
            }
        else:
            return {
                "membrane_potential": self.v,
                "spike_indices": self.spike_indices,
                "spike_times": self.spike_times,
                "total_spikes": self.total_spikes,
                "metrics": self.metrics,
            }


class MultiGPUNeuronSystem:
    """
    Multi-GPU system for scaling neuromorphic networks across multiple GPUs.
    Enables networks beyond single GPU memory limits.
    """
    
    def __init__(self, num_gpus: int = None, neuron_type: str = "adex"):
        """
        Initialize multi-GPU neuron system.
        
        Args:
            num_gpus: Number of GPUs to use (None = auto-detect)
            neuron_type: Type of neurons to simulate
        """
        self.neuron_type = neuron_type
        self.gpu_pools = []
        
        # Detect available GPUs
        if GPU_AVAILABLE:
            self.num_gpus = min(num_gpus or cp.cuda.runtime.getDeviceCount(), cp.cuda.runtime.getDeviceCount())
        else:
            self.num_gpus = 0
            
        print(f"MultiGPU System: {self.num_gpus} GPUs available")
        
    def distribute_neurons(self, total_neurons: int, **kwargs) -> List[GPUNeuronPool]:
        """
        Distribute neurons across available GPUs.
        
        Args:
            total_neurons: Total number of neurons to distribute
            **kwargs: Neuron parameters
            
        Returns:
            List of GPU neuron pools
        """
        if self.num_gpus == 0:
            # Fallback to CPU
            pool = GPUNeuronPool(total_neurons, self.neuron_type, use_gpu=False, **kwargs)
            return [pool]
        
        neurons_per_gpu = total_neurons // self.num_gpus
        remainder = total_neurons % self.num_gpus
        
        for gpu_id in range(self.num_gpus):
            with cp.cuda.Device(gpu_id):
                # Give remainder neurons to last GPU
                gpu_neurons = neurons_per_gpu + (remainder if gpu_id == self.num_gpus - 1 else 0)
                
                pool = GPUNeuronPool(
                    gpu_neurons, 
                    self.neuron_type, 
                    use_gpu=True,
                    **kwargs
                )
                self.gpu_pools.append(pool)
                
                print(f"GPU {gpu_id}: {gpu_neurons:,} neurons")
        
        return self.gpu_pools
    
    def synchronize_all(self):
        """Synchronize all GPU operations."""
        if GPU_AVAILABLE:
            for gpu_id in range(self.num_gpus):
                with cp.cuda.Device(gpu_id):
                    cp.cuda.Stream.null.synchronize()
    
    def get_total_memory_usage(self) -> float:
        """Get total memory usage across all GPUs in MB."""
        total_mb = 0.0
        
        if GPU_AVAILABLE:
            for gpu_id in range(self.num_gpus):
                with cp.cuda.Device(gpu_id):
                    mempool = cp.get_default_memory_pool()
                    total_mb += mempool.used_bytes() / (1024 ** 2)
        
        return total_mb
    
    def cleanup_all_memory(self):
        """Clean up memory on all GPUs."""
        for gpu_id in range(self.num_gpus):
            if GPU_AVAILABLE:
                with cp.cuda.Device(gpu_id):
                    cp.get_default_memory_pool().free_all_blocks()
                    cp.get_default_pinned_memory_pool().free_all_blocks()


class AdaptiveGPUNeuronPool(GPUNeuronPool):
    """
    Adaptive GPU neuron pool with dynamic optimization for large-scale networks.
    Integrates with GPUMemoryManager for optimal performance.
    """
    
    def __init__(self, num_neurons: int, **kwargs):
        """
        Initialize adaptive GPU neuron pool with automatic optimization.
        
        Args:
            num_neurons: Number of neurons
            **kwargs: Additional neuron parameters
        """
        # Import here to avoid circular dependency
        from .gpu_scaling import GPUMemoryManager
        
        self.memory_manager = GPUMemoryManager()
        
        # Get optimized configuration
        config = self.memory_manager.optimize_network_configuration(
            num_neurons, 
            kwargs.get('connectivity', 0.05)
        )
        
        # Use optimized parameters
        optimized_neurons = config['num_neurons']
        optimized_batch_size = config['batch_size']
        
        if config['optimization_applied']:
            print(f"GPU Optimization: {optimized_neurons:,} neurons (from {num_neurons:,})")
        
        # Initialize with optimized parameters
        super().__init__(
            optimized_neurons,
            batch_size=optimized_batch_size,
            **kwargs
        )
        
        self.optimization_config = config
        self.performance_history = []
        
    def adaptive_step(self, dt: float, I_syn: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """
        Adaptive step with performance monitoring and optimization.
        
        Args:
            dt: Time step
            I_syn: Synaptic currents
            
        Returns:
            Tuple of (spike_indices, enhanced_metrics)
        """
        # Monitor memory before step
        memory_before = self.memory_manager.monitor_memory_usage()
        
        # Perform regular step
        spike_indices, metrics = self.step(dt, I_syn)
        
        # Monitor memory after step
        memory_after = self.memory_manager.monitor_memory_usage()
        
        # Enhanced metrics with memory tracking
        enhanced_metrics = {
            **metrics,
            'memory_usage_mb': memory_after['used_mb'],
            'memory_percent': memory_after['percent_used'],
            'memory_delta_mb': memory_after['used_mb'] - memory_before['used_mb'],
            'optimization_config': self.optimization_config
        }
        
        # Track performance history
        self.performance_history.append(enhanced_metrics)
        
        # Adaptive memory cleanup (every 1000 steps)
        if len(self.performance_history) % 1000 == 0:
            if memory_after['percent_used'] > 85:  # High memory usage
                print(f"High GPU memory usage ({memory_after['percent_used']:.1f}%) - cleaning up")
                self._compact_memory()
        
        return spike_indices, enhanced_metrics
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.performance_history:
            return {}
        
        # Calculate statistics over performance history
        compute_times = [m['compute_time'] for m in self.performance_history]
        throughputs = [m['throughput'] for m in self.performance_history]
        memory_usage = [m['memory_usage_mb'] for m in self.performance_history]
        
        return {
            'total_steps': len(self.performance_history),
            'mean_compute_time': np.mean(compute_times),
            'std_compute_time': np.std(compute_times),
            'mean_throughput': np.mean(throughputs),
            'max_throughput': np.max(throughputs),
            'mean_memory_mb': np.mean(memory_usage),
            'peak_memory_mb': np.max(memory_usage),
            'total_spikes': self.total_spikes,
            'neurons': self.num_neurons,
            'optimization_applied': self.optimization_config.get('optimization_applied', False)
        }


# Convenience functions for large-scale GPU networks
def create_large_scale_gpu_network(
    target_neurons: int,
    neuron_type: str = "adex",
    use_adaptive: bool = True,
    **kwargs
) -> Union[GPUNeuronPool, AdaptiveGPUNeuronPool]:
    """
    Create a large-scale GPU-optimized neuron network.
    
    Args:
        target_neurons: Target number of neurons
        neuron_type: Type of neurons
        use_adaptive: Whether to use adaptive optimization
        **kwargs: Additional parameters
        
    Returns:
        GPU neuron pool instance
    """
    if not GPU_AVAILABLE:
        print("WARNING: GPU not available, falling back to CPU (will be slow!)")
        return GPUNeuronPool(min(target_neurons, 50000), neuron_type, use_gpu=False, **kwargs)
    
    if use_adaptive and target_neurons >= 10000:
        return AdaptiveGPUNeuronPool(target_neurons, neuron_type=neuron_type, **kwargs)
    else:
        return GPUNeuronPool(target_neurons, neuron_type, **kwargs)


def benchmark_gpu_scaling(max_neurons: int = 1000000) -> Dict[str, Any]:
    """
    Benchmark GPU scaling capabilities.
    
    Args:
        max_neurons: Maximum neurons to test
        
    Returns:
        Benchmark results
    """
    if not GPU_AVAILABLE:
        return {'error': 'GPU not available for benchmarking'}
    
    from .gpu_scaling import LargeScaleNetworkBuilder
    
    builder = LargeScaleNetworkBuilder()
    return builder.benchmark_network_size(max_neurons=max_neurons)


# Test function to verify GPU scaling
def test_gpu_scaling_limits():
    """
    Test GPU scaling limits with current hardware.
    """
    print("\n=== GPU Scaling Test ===")
    
    if not GPU_AVAILABLE:
        print("GPU not available - cannot test scaling")
        return
    
    # Test different network sizes
    test_sizes = [10000, 50000, 100000, 250000, 500000, 1000000]
    
    for size in test_sizes:
        try:
            print(f"Testing {size:,} neurons...")
            start_time = time.time()
            
            pool = create_large_scale_gpu_network(size, "adex")
            creation_time = time.time() - start_time
            
            # Test a few simulation steps
            for _ in range(10):
                pool.step(0.1)
            
            step_time = time.time() - start_time - creation_time
            
            print(f"  ✓ Success: {creation_time:.2f}s creation, {step_time:.2f}s simulation")
            
            # Clean up
            pool.clear_gpu_memory()
            
        except Exception as e:
            print(f"  ✗ Failed at {size:,} neurons: {str(e)}")
            break
    
    print("GPU scaling test completed.")
    Manages multiple GPU neuron pools for massive scale simulations.
    """

    def __init__(
        self,
        total_neurons: int,
        neurons_per_gpu: int = 100000,
        neuron_types: Optional[List[str]] = None,
    ):
        """
        Initialize multi-GPU neuron system.

        Args:
            total_neurons: Total number of neurons to simulate
            neurons_per_gpu: Maximum neurons per GPU
            neuron_types: List of neuron types to use
        """
        self.total_neurons = total_neurons
        self.neurons_per_gpu = neurons_per_gpu
        self.neuron_types = neuron_types or ["adex", "lif", "izhikevich"]

        # Calculate number of pools needed
        self.num_pools = (total_neurons + neurons_per_gpu - 1) // neurons_per_gpu

        # Initialize neuron pools
        self.pools = []
        remaining_neurons = total_neurons

        for i in range(self.num_pools):
            pool_size = min(neurons_per_gpu, remaining_neurons)
            neuron_type = self.neuron_types[i % len(self.neuron_types)]

            pool = GPUNeuronPool(
                num_neurons=pool_size,
                neuron_type=neuron_type,
                use_gpu=True,
                batch_size=10000,
            )
            self.pools.append(pool)
            remaining_neurons -= pool_size

        print(f"\nMulti-GPU System Initialized:")
        print(f"  Total neurons: {total_neurons:,}")
        print(f"  Number of pools: {self.num_pools}")
        print(f"  Neurons per pool: {neurons_per_gpu:,}")

    def simulate(self, duration: float, dt: float = 0.1) -> Dict:
        """
        Run simulation across all GPU pools.

        Args:
            duration: Simulation duration in milliseconds
            dt: Time step in milliseconds

        Returns:
            Dictionary with simulation results
        """
        num_steps = int(duration / dt)
        all_metrics = []

        print(f"\nStarting simulation: {duration}ms, {num_steps} steps")
        start_time = time.time()

        for step in range(num_steps):
            if step % 100 == 0:
                print(
                    f"  Step {step}/{num_steps} ({100*step/num_steps:.1f}%)", end="\r"
                )

            step_metrics = []
            for pool in self.pools:
                # Generate random input current
                I_syn = np.random.randn(pool.num_neurons) * 10

                # Step simulation
                spikes, metrics = pool.step(dt, I_syn)
                step_metrics.append(metrics)

            all_metrics.append(step_metrics)

        total_time = time.time() - start_time
        print(f"\nSimulation completed in {total_time:.2f}s")

        # Aggregate results
        results = self._aggregate_results(all_metrics, total_time)
        return results

    def _aggregate_results(self, all_metrics: List, total_time: float) -> Dict:
        """Aggregate results from all pools."""
        total_spikes = sum(pool.total_spikes for pool in self.pools)

        # Get statistics from each pool
        pool_stats = [pool.get_spike_statistics() for pool in self.pools]

        # Calculate aggregate statistics
        results = {
            "total_neurons": self.total_neurons,
            "total_spikes": total_spikes,
            "simulation_time": total_time,
            "neurons_per_second": self.total_neurons / total_time,
            "spikes_per_second": total_spikes / total_time,
            "num_pools": self.num_pools,
            "pool_statistics": pool_stats,
        }

        # Add GPU memory usage
        if GPU_AVAILABLE:
            total_gpu_memory = sum(
                pool.metrics[-1]["gpu_memory_used_mb"]
                for pool in self.pools
                if pool.metrics
            )
            results["total_gpu_memory_mb"] = total_gpu_memory

        return results

    def cleanup(self):
        """Clean up GPU resources."""
        for pool in self.pools:
            pool.clear_gpu_memory()
        print("All GPU resources cleaned up")


def analyze_gpu_performance(num_neurons_list: List[int]) -> Dict:
    """
    Analyze GPU performance across different scales.

    Args:
        num_neurons_list: List of neuron counts to test

    Returns:
        Performance analysis results
    """
    results = {}

    for num_neurons in num_neurons_list:
        print(f"\n{'='*60}")
        print(f"Testing {num_neurons:,} neurons")
        print("=" * 60)

        try:
            # Create GPU neuron pool
            pool = GPUNeuronPool(
                num_neurons=num_neurons,
                neuron_type="adex",
                use_gpu=True,
                batch_size=min(10000, num_neurons),
            )

            # Run simulation
            duration = 100.0  # 100ms
            dt = 0.1
            num_steps = int(duration / dt)

            step_times = []
            for step in range(num_steps):
                I_syn = np.random.randn(num_neurons) * 10
                start = time.time()
                spikes, metrics = pool.step(dt, I_syn)
                step_times.append(time.time() - start)

            # Get statistics
            stats = pool.get_spike_statistics()

            # Store results
            results[num_neurons] = {
                "mean_step_time": np.mean(step_times),
                "std_step_time": np.std(step_times),
                "min_step_time": np.min(step_times),
                "max_step_time": np.max(step_times),
                "total_spikes": stats["total_spikes"],
                "active_neurons": stats["active_neurons"],
                "neurons_per_second": num_neurons / np.mean(step_times),
                "gpu_memory_mb": (
                    pool.metrics[-1]["gpu_memory_used_mb"] if pool.metrics else 0
                ),
            }

            # Print summary
            print(f"Success:")
            print(
                f"  Mean step time: {results[num_neurons]['mean_step_time']*1000:.2f}ms"
            )
            print(
                f"  Neurons/second: {results[num_neurons]['neurons_per_second']:,.0f}"
            )
            print(f"  Total spikes: {results[num_neurons]['total_spikes']:,}")
            print(f"  GPU memory: {results[num_neurons]['gpu_memory_mb']:.1f}MB")

            # Cleanup
            pool.clear_gpu_memory()

        except Exception as e:
            print(f"[FAILED] Failed: {e}")
            results[num_neurons] = {"error": str(e)}

    return results


if __name__ == "__main__":
    print("GPU-Accelerated Neuron System Demo")
    print("=" * 60)

    # Test different scales
    test_scales = [10000, 50000, 100000, 500000, 1000000]

    print("\n1. Testing single GPU pool scaling:")
    performance_results = analyze_gpu_performance(test_scales[:3])

    print("\n2. Testing multi-GPU system:")
    multi_system = MultiGPUNeuronSystem(total_neurons=1000000, neurons_per_gpu=100000)

    # Run short simulation
    results = multi_system.simulate(duration=50.0, dt=0.1)

    print("\nMulti-GPU Simulation Results:")
    print(f"  Total neurons: {results['total_neurons']:,}")
    print(f"  Total spikes: {results['total_spikes']:,}")
    print(f"  Simulation time: {results['simulation_time']:.2f}s")
    print(f"  Neurons/second: {results['neurons_per_second']:,.0f}")
    print(f"  Spikes/second: {results['spikes_per_second']:,.0f}")

    if "total_gpu_memory_mb" in results:
        print(f"  Total GPU memory: {results['total_gpu_memory_mb']:.1f}MB")

    # Cleanup
    multi_system.cleanup()
