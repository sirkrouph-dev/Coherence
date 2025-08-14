#!/usr/bin/env python3
"""
Find Maximum GPU Learning Capacity
==================================

Systematically test to find the maximum number of neurons we can handle
with full learning capability on your RTX 3060 (8GB VRAM).
"""

import numpy as np
import psutil
import time
from typing import Tuple

# Import our learning system
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cupy as cp
    from cupyx.scipy import sparse as cp_sparse
    GPU_AVAILABLE = True
    print(f"[OK] CuPy GPU acceleration available")
    print(f"GPU Memory: {cp.cuda.Device().mem_info[1] / 1024**3:.1f}GB total")
except ImportError:
    GPU_AVAILABLE = False
    print("[WARNING] No GPU available")
    exit(1)

class MemoryTracker:
    """Track GPU and system memory usage"""
    
    def __init__(self):
        self.gpu_device = cp.cuda.Device()
        self.process = psutil.Process()
    
    def get_gpu_memory_mb(self) -> Tuple[float, float]:
        """Returns (used_mb, total_mb)"""
        meminfo = self.gpu_device.mem_info
        return meminfo[1] / 1024**2, meminfo[0] / 1024**2
    
    def get_system_memory_mb(self) -> float:
        """Returns system memory used by this process in MB"""
        return self.process.memory_info().rss / 1024**2
    
    def print_status(self, label: str = ""):
        gpu_used, gpu_total = self.get_gpu_memory_mb()
        sys_used = self.get_system_memory_mb()
        print(f"{label:20} GPU: {gpu_used:6.1f}MB/{gpu_total:6.1f}MB | System: {sys_used:6.1f}MB")

class QuickLearningTest:
    """Minimal learning system for capacity testing"""
    
    def __init__(self, num_neurons: int, connectivity: float = 0.001):
        self.num_neurons = num_neurons
        self.connectivity = connectivity
        self.num_connections = int(num_neurons * num_neurons * connectivity)
        
        print(f"Testing {num_neurons:,} neurons, {self.num_connections:,} synapses")
        
        # Neuron states (minimal)
        self.membrane_potential = cp.zeros(num_neurons, dtype=cp.float32)
        self.spike_times = cp.zeros(num_neurons, dtype=cp.float32)
        
        # Sparse synaptic weights
        if self.num_connections > 0:
            self.pre_indices = cp.random.randint(0, num_neurons, self.num_connections).astype(cp.int32)
            self.post_indices = cp.random.randint(0, num_neurons, self.num_connections).astype(cp.int32)
            self.weights = cp.random.uniform(0.0, 0.1, self.num_connections).astype(cp.float32)
            
            # STDP traces
            self.pre_trace = cp.zeros(num_neurons, dtype=cp.float32)
            self.post_trace = cp.zeros(num_neurons, dtype=cp.float32)
    
    def step_simulation(self, dt: float = 0.001, steps: int = 100):
        """Run a few simulation steps to test learning"""
        for i in range(steps):
            # Simple LIF dynamics
            I_input = cp.random.randn(self.num_neurons) * 5
            
            # Membrane potential update
            self.membrane_potential += dt * (I_input - self.membrane_potential) / 0.02
            
            # Spike detection
            spikes = self.membrane_potential > 1.0
            spike_indices = cp.where(spikes)[0]
            
            # Reset spiked neurons
            self.membrane_potential[spikes] = 0.0
            
            # STDP learning (if we have synapses)
            if self.num_connections > 0:
                # Update traces
                self.pre_trace *= 0.99
                self.post_trace *= 0.99
                self.pre_trace[spike_indices] = 1.0
                self.post_trace[spike_indices] = 1.0
                
                # Weight updates (simplified STDP)
                if len(spike_indices) > 0:
                    # Potentiation
                    pre_active = cp.isin(self.pre_indices, spike_indices)
                    self.weights[pre_active] += 0.001 * self.post_trace[self.post_indices[pre_active]]
                    
                    # Depression  
                    post_active = cp.isin(self.post_indices, spike_indices)
                    self.weights[post_active] -= 0.001 * self.pre_trace[self.pre_indices[post_active]]
                    
                    # Keep weights positive
                    self.weights = cp.maximum(self.weights, 0.0)
        
        return len(spike_indices)

def test_learning_capacity():
    """Find maximum learning capacity through binary search"""
    print("🔍 FINDING MAXIMUM GPU LEARNING CAPACITY")
    print("=" * 60)
    
    tracker = MemoryTracker()
    tracker.print_status("Initial")
    
    # Binary search parameters
    min_neurons = 850_000  # We know this works
    max_neurons = 10_000_000  # Start optimistically
    best_working = min_neurons
    
    while min_neurons <= max_neurons:
        test_neurons = (min_neurons + max_neurons) // 2
        print(f"\n🧠 Testing {test_neurons:,} neurons...")
        
        try:
            # Clear GPU memory
            cp.get_default_memory_pool().free_all_blocks()
            tracker.print_status("Before test")
            
            # Create test network
            start_time = time.time()
            network = QuickLearningTest(test_neurons, connectivity=0.0005)  # Reduce connectivity for more neurons
            creation_time = time.time() - start_time
            
            tracker.print_status("After creation")
            
            # Test learning
            start_time = time.time()
            spikes = network.step_simulation(steps=50)
            learning_time = time.time() - start_time
            
            tracker.print_status("After learning")
            
            print(f"✅ SUCCESS: {test_neurons:,} neurons")
            print(f"   Creation: {creation_time:.2f}s")
            print(f"   Learning: {learning_time:.2f}s") 
            print(f"   Spikes: {spikes}")
            
            best_working = test_neurons
            min_neurons = test_neurons + 1
            
            # Clean up
            del network
            cp.get_default_memory_pool().free_all_blocks()
            
        except Exception as e:
            print(f"❌ FAILED: {test_neurons:,} neurons")
            print(f"   Error: {str(e)[:100]}...")
            max_neurons = test_neurons - 1
            
            # Clean up on failure
            cp.get_default_memory_pool().free_all_blocks()
    
    print(f"\n🎉 MAXIMUM CAPACITY FOUND: {best_working:,} neurons")
    
    # Final detailed test of maximum capacity
    print(f"\n🔬 DETAILED TEST OF MAXIMUM CAPACITY")
    print("=" * 40)
    
    try:
        cp.get_default_memory_pool().free_all_blocks()
        
        # Test with different connectivity levels
        connectivities = [0.001, 0.0005, 0.0001]
        
        for conn in connectivities:
            print(f"\nTesting {best_working:,} neurons with {conn:.4f} connectivity...")
            
            start_time = time.time()
            network = QuickLearningTest(best_working, connectivity=conn)
            
            # Run extended learning test
            total_spikes = 0
            for epoch in range(5):
                spikes = network.step_simulation(steps=100)
                total_spikes += spikes
                tracker.print_status(f"Epoch {epoch+1}")
            
            total_time = time.time() - start_time
            
            print(f"✅ Connectivity {conn:.4f}: {total_spikes} total spikes in {total_time:.2f}s")
            print(f"   Performance: {best_working * 5 * 100 / total_time:.0f} neuron-steps/second")
            
            del network
            cp.get_default_memory_pool().free_all_blocks()
            
    except Exception as e:
        print(f"❌ Detailed test failed: {e}")
    
    return best_working

if __name__ == "__main__":
    max_capacity = test_learning_capacity()
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"Maximum GPU learning capacity: {max_capacity:,} neurons")
    print(f"RTX 3060 8GB utilization optimized!")
