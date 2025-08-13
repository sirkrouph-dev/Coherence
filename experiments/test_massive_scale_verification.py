#!/usr/bin/env python3
"""
Test massive scale performance to verify 50M+ neuron claims
"""

import numpy as np
import time
import psutil
import os
from core.gpu_neurons import GPUNeuronPool

def measure_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def test_massive_scale():
    """Test massive scale networks to verify our claims"""
    print("🎯 MASSIVE SCALE NEUROMORPHIC VERIFICATION")
    print("=" * 60)
    
    # Test progressively larger networks
    sizes = [1_000_000, 5_000_000, 10_000_000, 20_000_000, 50_000_000]
    
    for size in sizes:
        print(f"\n🧠 Testing {size:,} neurons...")
        
        try:
            start_mem = measure_memory_usage()
            
            # Create pool
            pool = GPUNeuronPool(size, 'lif')
            
            # Generate random synaptic input
            synaptic_input = np.random.randn(size) * 5
            
            after_creation_mem = measure_memory_usage()
            creation_memory = after_creation_mem - start_mem
            
            # Run simulation steps
            n_steps = 5  # Fewer steps for massive networks
            total_spikes = 0
            
            print(f"  Running {n_steps} simulation steps...")
            start_time = time.time()
            
            for step in range(n_steps):
                spike_indices, metrics = pool.step(0.1, synaptic_input)
                total_spikes += len(spike_indices)
                print(f"    Step {step+1}: {len(spike_indices):,} spikes")
            
            end_time = time.time()
            final_mem = measure_memory_usage()
            
            # Calculate metrics
            sim_time = end_time - start_time
            total_memory = final_mem - start_mem
            throughput = (size * n_steps) / sim_time
            avg_step_time = sim_time / n_steps * 1000  # ms per step
            memory_per_neuron = total_memory * 1024 / size  # KB per neuron
            spike_rate = total_spikes / (size * n_steps * 0.1) * 1000  # Hz
            
            print(f"  ✅ SUCCESS!")
            print(f"    Total time:        {sim_time*1000:.1f}ms")
            print(f"    Avg time per step: {avg_step_time:.1f}ms")
            print(f"    Throughput:        {throughput:,.0f} neurons/second")
            print(f"    Total memory:      {total_memory:.1f}MB")
            print(f"    Memory per neuron: {memory_per_neuron:.1f} KB")
            print(f"    Total spikes:      {total_spikes:,}")
            print(f"    Spike rate:        {spike_rate:.1f} Hz")
            
            # Performance analysis
            neurons_per_ms = size / avg_step_time * 1000
            print(f"    Performance:       {neurons_per_ms:,.0f} neurons/ms")
            
            if throughput > 1_000_000_000:  # 1 billion neurons/second
                print(f"    🏆 BILLION+ NEURONS/SECOND ACHIEVED!")
            
            # Cleanup
            del pool
            
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"🎉 MASSIVE SCALE VERIFICATION COMPLETE")

def test_50m_detailed():
    """Detailed test of 50M neurons with multiple steps"""
    print(f"\n🔬 DETAILED 50M NEURON TEST")
    print("=" * 40)
    
    size = 50_000_000
    n_steps = 10
    
    try:
        start_mem = measure_memory_usage()
        
        print(f"Creating {size:,} neuron pool...")
        pool = GPUNeuronPool(size, 'lif')
        
        creation_mem = measure_memory_usage()
        print(f"Memory after creation: {creation_mem - start_mem:.1f}MB")
        
        # Test with different input patterns
        patterns = {
            "random": np.random.randn(size) * 3,
            "sparse": np.zeros(size),
            "strong": np.full(size, 15.0)
        }
        
        patterns["sparse"][:size//1000] = 20.0  # 0.1% active
        
        for pattern_name, synaptic_input in patterns.items():
            print(f"\n  Testing with {pattern_name} input pattern:")
            
            total_spikes = 0
            step_times = []
            
            for step in range(n_steps):
                step_start = time.time()
                spike_indices, metrics = pool.step(0.1, synaptic_input)
                step_end = time.time()
                
                step_time = (step_end - step_start) * 1000  # ms
                step_times.append(step_time)
                num_spikes = len(spike_indices)
                total_spikes += num_spikes
                
                if step < 3:  # Show first few steps
                    print(f"    Step {step+1}: {step_time:.1f}ms, {num_spikes:,} spikes")
            
            # Statistics
            avg_step_time = np.mean(step_times)
            throughput = size / (avg_step_time / 1000)
            spike_rate = total_spikes / (size * n_steps * 0.1) * 1000
            
            print(f"    Average step time: {avg_step_time:.1f}ms")
            print(f"    Throughput:        {throughput:,.0f} neurons/second")
            print(f"    Total spikes:      {total_spikes:,}")
            print(f"    Spike rate:        {spike_rate:.1f} Hz")
        
        final_mem = measure_memory_usage()
        print(f"\n  Final memory usage: {final_mem - start_mem:.1f}MB")
        print(f"  Memory efficiency:  {(final_mem - start_mem)*1024/size:.1f} KB/neuron")
        
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_massive_scale()
    test_50m_detailed()
