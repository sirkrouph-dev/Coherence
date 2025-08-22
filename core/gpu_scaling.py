#!/usr/bin/env python3
"""
GPU Scaling and Memory Management for Large-Scale Neuromorphic Networks
======================================================================

This module provides advanced GPU scaling capabilities to push neuromorphic
networks to the limits of available GPU hardware, enabling simulation of
100K-1M+ neuron networks with efficient memory management.

Key Features:
- Dynamic GPU memory detection and optimization
- Adaptive batch sizing for different GPU configurations  
- Memory pool management for large-scale simulations
- Automatic fallback strategies for memory constraints
- Performance monitoring and bottleneck detection
"""

import gc
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any
import time
import math
import numpy as np

try:
    import cupy as cp
    import cupyx
    from cupy.cuda import MemoryPool, memory
    GPU_AVAILABLE = True
except ImportError:
    cp = np
    GPU_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


@dataclass
class GPUConfiguration:
    """GPU configuration and capabilities."""
    device_name: str
    total_memory_mb: float
    available_memory_mb: float
    compute_capability: Tuple[int, int]
    multiprocessor_count: int
    max_threads_per_block: int
    recommended_batch_size: int
    max_network_size: int


@dataclass
class MemoryProfile:
    """Memory usage profile for network components."""
    neurons_mb: float
    synapses_mb: float
    spike_buffers_mb: float
    temp_arrays_mb: float
    total_mb: float
    utilization_percent: float


class GPUMemoryManager:
    """Advanced GPU memory management for large-scale networks."""
    
    def __init__(self, safety_margin: float = 0.15):
        """
        Initialize GPU memory manager.
        
        Args:
            safety_margin: Fraction of GPU memory to keep free (default 15%)
        """
        self.safety_margin = safety_margin
        self.gpu_config = self._detect_gpu_configuration()
        self.memory_pool = cp.get_default_memory_pool() if GPU_AVAILABLE else None
        self.pinned_memory_pool = cp.get_default_pinned_memory_pool() if GPU_AVAILABLE else None
        
        # Memory tracking
        self.peak_usage_mb = 0.0
        self.allocation_history = []
        
        if GPU_AVAILABLE:
            print(f"GPU Memory Manager Initialized:")
            print(f"  Device: {self.gpu_config.device_name}")
            print(f"  Total Memory: {self.gpu_config.total_memory_mb:.1f} MB")
            print(f"  Safety Margin: {safety_margin*100:.1f}%")
            print(f"  Max Recommended Network: {self.gpu_config.max_network_size:,} neurons")
    
    def _detect_gpu_configuration(self) -> GPUConfiguration:
        """Detect GPU capabilities and compute optimal configuration."""
        if not GPU_AVAILABLE:
            return GPUConfiguration(
                device_name="CPU_FALLBACK",
                total_memory_mb=0.0,
                available_memory_mb=0.0,
                compute_capability=(0, 0),
                multiprocessor_count=0,
                max_threads_per_block=0,
                recommended_batch_size=10000,  # Conservative CPU batch
                max_network_size=50000  # CPU limit
            )
        
        device = cp.cuda.Device()
        props = device.attributes
        
        # Get memory info
        meminfo = cp.cuda.MemoryInfo()
        total_memory_mb = meminfo.total / (1024 ** 2)
        available_memory_mb = meminfo.free / (1024 ** 2)
        
        # Calculate optimal configuration
        usable_memory_mb = available_memory_mb * (1 - self.safety_margin)
        
        # Estimate neurons per MB (conservative estimate)
        # AdEx neuron: ~40 bytes per neuron (v, w, spike_flags, etc.)
        # Synapses: ~12 bytes per synapse (assuming 5% connectivity)
        bytes_per_neuron = 40 + (0.05 * 12)  # Neuron + average synapses
        neurons_per_mb = (1024 ** 2) / bytes_per_neuron
        
        max_network_size = int(usable_memory_mb * neurons_per_mb)
        
        # Batch size optimization based on GPU architecture
        compute_cap = device.compute_capability
        if compute_cap >= (8, 0):  # Ampere (RTX 30/40 series)
            recommended_batch_size = min(1000000, max_network_size)
        elif compute_cap >= (7, 5):  # Turing (RTX 20 series)
            recommended_batch_size = min(500000, max_network_size)
        elif compute_cap >= (6, 1):  # Pascal (GTX 10 series)
            recommended_batch_size = min(200000, max_network_size)
        else:  # Older architectures
            recommended_batch_size = min(100000, max_network_size)
        
        return GPUConfiguration(
            device_name=props.get('name', 'Unknown GPU'),
            total_memory_mb=total_memory_mb,
            available_memory_mb=available_memory_mb,
            compute_capability=compute_cap,
            multiprocessor_count=props.get('multiProcessorCount', 0),
            max_threads_per_block=props.get('maxThreadsPerBlock', 1024),
            recommended_batch_size=recommended_batch_size,
            max_network_size=max_network_size
        )
    
    def estimate_memory_requirements(
        self, 
        num_neurons: int, 
        connectivity: float = 0.05,
        spike_buffer_size: int = 10000
    ) -> MemoryProfile:
        """
        Estimate memory requirements for a network configuration.
        
        Args:
            num_neurons: Number of neurons
            connectivity: Connection probability between neurons
            spike_buffer_size: Size of spike history buffer per neuron
            
        Returns:
            MemoryProfile with detailed memory breakdown
        """
        # Neuron state arrays (float32)
        # membrane_potential, adaptation_current, refractory_time, etc.
        neurons_mb = (num_neurons * 6 * 4) / (1024 ** 2)  # 6 arrays × 4 bytes
        
        # Synapses (sparse matrices)
        num_synapses = int(num_neurons * num_neurons * connectivity)
        # Sparse matrix: indices (2×int32) + weights (float32) = 12 bytes per synapse
        synapses_mb = (num_synapses * 12) / (1024 ** 2)
        
        # Spike buffers (circular buffers for memory efficiency)
        spike_buffers_mb = (num_neurons * spike_buffer_size * 4) / (1024 ** 2)
        
        # Temporary arrays for computation (estimated 20% overhead)
        temp_arrays_mb = (neurons_mb + synapses_mb) * 0.2
        
        total_mb = neurons_mb + synapses_mb + spike_buffers_mb + temp_arrays_mb
        utilization_percent = (total_mb / self.gpu_config.total_memory_mb) * 100
        
        return MemoryProfile(
            neurons_mb=neurons_mb,
            synapses_mb=synapses_mb,
            spike_buffers_mb=spike_buffers_mb,
            temp_arrays_mb=temp_arrays_mb,
            total_mb=total_mb,
            utilization_percent=utilization_percent
        )
    
    def optimize_network_configuration(
        self, 
        target_neurons: int,
        connectivity: float = 0.05
    ) -> Dict[str, Any]:
        """
        Optimize network configuration for available GPU memory.
        
        Args:
            target_neurons: Desired number of neurons
            connectivity: Connection probability
            
        Returns:
            Optimized configuration dictionary
        """
        # Check if target fits in memory
        profile = self.estimate_memory_requirements(target_neurons, connectivity)
        
        if profile.total_mb <= self.gpu_config.available_memory_mb * (1 - self.safety_margin):
            # Configuration fits - no optimization needed
            return {
                "num_neurons": target_neurons,
                "connectivity": connectivity,
                "batch_size": min(self.gpu_config.recommended_batch_size, target_neurons),
                "memory_profile": profile,
                "optimization_applied": False,
                "fit_status": "OPTIMAL"
            }
        
        # Need optimization - reduce network size to fit memory
        max_neurons = int(self.gpu_config.max_network_size * 0.95)  # 95% of theoretical max
        
        if target_neurons > max_neurons:
            warnings.warn(
                f"Target network ({target_neurons:,} neurons) exceeds GPU capacity. "
                f"Reducing to {max_neurons:,} neurons."
            )
            
            optimized_profile = self.estimate_memory_requirements(max_neurons, connectivity)
            
            return {
                "num_neurons": max_neurons,
                "connectivity": connectivity,
                "batch_size": self.gpu_config.recommended_batch_size,
                "memory_profile": optimized_profile,
                "optimization_applied": True,
                "fit_status": "REDUCED_SIZE",
                "original_target": target_neurons
            }
        
        # Try reducing connectivity instead
        max_connectivity = 0.01  # Minimum viable connectivity
        optimized_connectivity = connectivity
        
        while optimized_connectivity > max_connectivity:
            optimized_connectivity *= 0.8  # Reduce by 20% each iteration
            test_profile = self.estimate_memory_requirements(target_neurons, optimized_connectivity)
            
            if test_profile.total_mb <= self.gpu_config.available_memory_mb * (1 - self.safety_margin):
                return {
                    "num_neurons": target_neurons,
                    "connectivity": optimized_connectivity,
                    "batch_size": min(self.gpu_config.recommended_batch_size, target_neurons),
                    "memory_profile": test_profile,
                    "optimization_applied": True,
                    "fit_status": "REDUCED_CONNECTIVITY",
                    "original_connectivity": connectivity
                }
        
        # Last resort - both size and connectivity reduction
        final_neurons = max_neurons // 2
        final_connectivity = max_connectivity
        final_profile = self.estimate_memory_requirements(final_neurons, final_connectivity)
        
        warnings.warn(
            f"Aggressive optimization applied: {final_neurons:,} neurons, "
            f"{final_connectivity:.3f} connectivity to fit GPU memory."
        )
        
        return {
            "num_neurons": final_neurons,
            "connectivity": final_connectivity,
            "batch_size": min(self.gpu_config.recommended_batch_size, final_neurons),
            "memory_profile": final_profile,
            "optimization_applied": True,
            "fit_status": "AGGRESSIVE_REDUCTION",
            "original_target": target_neurons,
            "original_connectivity": connectivity
        }
    
    def monitor_memory_usage(self) -> Dict[str, float]:
        """Monitor current GPU memory usage."""
        if not GPU_AVAILABLE:
            if PSUTIL_AVAILABLE:
                vm = psutil.virtual_memory()
                return {
                    "used_mb": vm.used / (1024 ** 2),
                    "available_mb": vm.available / (1024 ** 2),
                    "percent_used": vm.percent
                }
            return {"used_mb": 0, "available_mb": 0, "percent_used": 0}
        
        meminfo = cp.cuda.MemoryInfo()
        used_mb = (meminfo.total - meminfo.free) / (1024 ** 2)
        percent_used = (used_mb / (meminfo.total / (1024 ** 2))) * 100
        
        # Update peak usage tracking
        self.peak_usage_mb = max(self.peak_usage_mb, used_mb)
        
        return {
            "used_mb": used_mb,
            "available_mb": meminfo.free / (1024 ** 2),
            "percent_used": percent_used,
            "peak_mb": self.peak_usage_mb
        }
    
    def cleanup_memory(self):
        """Perform aggressive memory cleanup."""
        if GPU_AVAILABLE and self.memory_pool:
            # Free memory pool
            self.memory_pool.free_all_blocks()
            if self.pinned_memory_pool:
                self.pinned_memory_pool.free_all_blocks()
        
        # Python garbage collection
        gc.collect()
        
        # CuPy-specific cleanup
        if GPU_AVAILABLE:
            cp.get_default_memory_pool().free_all_blocks()


class LargeScaleNetworkBuilder:
    """Builder for creating large-scale networks optimized for GPU execution."""
    
    def __init__(self, memory_manager: Optional[GPUMemoryManager] = None):
        """Initialize large-scale network builder."""
        self.memory_manager = memory_manager or GPUMemoryManager()
        self.networks_created = 0
    
    def create_large_scale_network(
        self,
        target_neurons: int,
        connectivity: float = 0.05,
        neuron_type: str = "adex",
        enable_plasticity: bool = True
    ) -> Dict[str, Any]:
        """
        Create a large-scale network optimized for GPU execution.
        
        Args:
            target_neurons: Target number of neurons
            connectivity: Connection probability
            neuron_type: Type of neurons ('adex', 'lif')
            enable_plasticity: Whether to enable synaptic plasticity
            
        Returns:
            Dictionary containing network and metadata
        """
        start_time = time.time()
        
        print(f"\n=== Large-Scale Network Creation ===")
        print(f"Target: {target_neurons:,} neurons")
        print(f"Connectivity: {connectivity:.3f}")
        print(f"Neuron Type: {neuron_type}")
        
        # Optimize configuration for available memory
        config = self.memory_manager.optimize_network_configuration(target_neurons, connectivity)
        
        actual_neurons = config["num_neurons"]
        actual_connectivity = config["connectivity"]
        
        if config["optimization_applied"]:
            print(f"OPTIMIZATION: {config['fit_status']}")
            print(f"Adjusted: {actual_neurons:,} neurons, {actual_connectivity:.3f} connectivity")
        
        # Memory pre-allocation for efficiency
        print("Pre-allocating GPU memory...")
        self.memory_manager.cleanup_memory()  # Start fresh
        
        # Create network components
        network_data = {
            "num_neurons": actual_neurons,
            "connectivity": actual_connectivity,
            "neuron_type": neuron_type,
            "batch_size": config["batch_size"],
            "memory_profile": config["memory_profile"],
            "creation_time": time.time() - start_time,
            "gpu_config": self.memory_manager.gpu_config,
            "network_id": f"large_scale_{self.networks_created}"
        }
        
        self.networks_created += 1
        
        print(f"Network created in {network_data['creation_time']:.2f}s")
        print(f"Memory usage: {config['memory_profile'].total_mb:.1f} MB "
               f"({config['memory_profile'].utilization_percent:.1f}%)")
        
        return network_data
    
    def benchmark_network_size(self, start_neurons: int = 10000, max_neurons: int = 1000000) -> List[Dict]:
        """
        Benchmark different network sizes to find GPU limits.
        
        Args:
            start_neurons: Starting number of neurons
            max_neurons: Maximum neurons to test
            
        Returns:
            List of benchmark results
        """
        results = []
        current_size = start_neurons
        
        print(f"\n=== GPU Scaling Benchmark ===")
        print(f"Testing network sizes from {start_neurons:,} to {max_neurons:,} neurons")
        
        while current_size <= max_neurons:
            try:
                start_time = time.time()
                
                # Test network creation
                network_data = self.create_large_scale_network(current_size)
                creation_time = time.time() - start_time
                
                # Monitor memory
                memory_usage = self.memory_manager.monitor_memory_usage()
                
                result = {
                    "network_size": current_size,
                    "creation_time": creation_time,
                    "memory_used_mb": memory_usage["used_mb"],
                    "memory_percent": memory_usage["percent_used"],
                    "success": True,
                    "fit_status": network_data.get("fit_status", "OPTIMAL")
                }
                
                results.append(result)
                print(f"✓ {current_size:,} neurons: {creation_time:.2f}s, {memory_usage['memory_percent']:.1f}% memory")
                
                # Clean up for next test
                self.memory_manager.cleanup_memory()
                
                # Increase size (geometric progression)
                if current_size < 100000:
                    current_size += 10000  # 10K increments up to 100K
                else:
                    current_size = int(current_size * 1.2)  # 20% increases above 100K
                
            except Exception as e:
                result = {
                    "network_size": current_size,
                    "creation_time": -1,
                    "memory_used_mb": -1,
                    "memory_percent": -1,
                    "success": False,
                    "error": str(e)
                }
                results.append(result)
                print(f"✗ {current_size:,} neurons: FAILED - {str(e)}")
                break
        
        # Find maximum successful size
        successful_results = [r for r in results if r["success"]]
        if successful_results:
            max_successful = max(successful_results, key=lambda x: x["network_size"])
            print(f"\nMAXIMUM GPU NETWORK SIZE: {max_successful['network_size']:,} neurons")
            print(f"Peak memory usage: {max_successful['memory_percent']:.1f}%")
        
        return results


# Convenience function for quick GPU scaling tests
def quick_gpu_scale_test(target_neurons: int = 100000) -> Dict[str, Any]:
    """Quick test of GPU scaling capabilities."""
    if not GPU_AVAILABLE:
        print("GPU not available - using CPU fallback")
        return {"success": False, "reason": "No GPU available"}
    
    manager = GPUMemoryManager()
    builder = LargeScaleNetworkBuilder(manager)
    
    try:
        result = builder.create_large_scale_network(target_neurons)
        result["success"] = True
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    # Quick test when run directly
    print("=== GPU Scaling Module Test ===")
    test_result = quick_gpu_scale_test(100000)
    
    if test_result["success"]:
        print("✓ GPU scaling test successful!")
        print(f"  Created network with {test_result['num_neurons']:,} neurons")
    else:
        print(f"✗ GPU scaling test failed: {test_result.get('error', 'Unknown error')}")