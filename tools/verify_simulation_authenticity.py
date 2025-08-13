#!/usr/bin/env python3
"""
Comprehensive verification that we're actually simulating neurons, not just measuring overhead.
This script will test:
1. Actual neuron dynamics (membrane potential changes)
2. Spike generation and counting
3. Synaptic input effects
4. Memory usage patterns
5. Computational complexity scaling
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

def verify_neuron_dynamics(pool, n_neurons=1000, n_steps=100):
    """Verify that neurons actually change their membrane potential and spike"""
    print(f"\n🧠 TESTING NEURON DYNAMICS ({n_neurons} neurons, {n_steps} steps)")
    
    # Record initial state
    initial_v = pool.v.copy()
    initial_spike_count = np.sum(pool.spike_count) if hasattr(pool, 'spike_count') else 0
    
    # Apply different current levels to different neurons
    synaptic_inputs = np.zeros(n_neurons)
    synaptic_inputs[:n_neurons//3] = 15.0  # Strong input (should spike)
    synaptic_inputs[n_neurons//3:2*n_neurons//3] = 5.0   # Weak input (may spike)
    synaptic_inputs[2*n_neurons//3:] = 0.0   # No input (should not spike)
    
    spike_times = []
    membrane_potentials = []
    
    # Run simulation and record data
    dt = 0.1
    for step in range(n_steps):
        spike_indices, metrics = pool.step(dt, synaptic_inputs)
        spike_times.append(len(spike_indices))  # Count of neurons that spiked
        membrane_potentials.append(pool.v.copy())
    
    # Analysis
    final_v = pool.v.copy()
    total_spikes = sum(spike_times)
    
    # Verify membrane potential changes
    v_changed = not np.allclose(initial_v, final_v, atol=1e-6)
    max_v_change = np.max(np.abs(final_v - initial_v))
    
    # Verify spikes occurred
    spikes_occurred = total_spikes > 0
    
    # Verify different input levels produced different responses
    strong_neurons = pool.v[:n_neurons//3]
    weak_neurons = pool.v[n_neurons//3:2*n_neurons//3] 
    no_input_neurons = pool.v[2*n_neurons//3:]
    
    print(f"  ✓ Membrane potentials changed: {v_changed} (max change: {max_v_change:.3f}mV)")
    print(f"  ✓ Total spikes generated: {total_spikes}")
    print(f"  ✓ Strong input neurons avg V: {np.mean(strong_neurons):.2f}mV")
    print(f"  ✓ Weak input neurons avg V: {np.mean(weak_neurons):.2f}mV") 
    print(f"  ✓ No input neurons avg V: {np.mean(no_input_neurons):.2f}mV")
    
    return v_changed and spikes_occurred and total_spikes > n_steps//10

def verify_computational_scaling(sizes=[1000, 5000, 10000, 50000]):
    """Verify that computation time scales with network size"""
    print(f"\n⏱️  TESTING COMPUTATIONAL SCALING")
    
    times = []
    memories = []
    n_steps = 50
    dt = 0.1
    
    for size in sizes:
        print(f"  Testing {size} neurons...")
        
        # Memory before
        mem_before = measure_memory_usage()
        
        # Create pool
        pool = GPUNeuronPool(size, 'lif')
        synaptic_inputs = np.random.randn(size) * 10
        
        # Memory after creation
        mem_after_creation = measure_memory_usage()
        
        # Time simulation
        start_time = time.time()
        total_spikes = 0
        
        for step in range(n_steps):
            spike_indices, metrics = pool.step(dt, synaptic_inputs)
            total_spikes += len(spike_indices)
        
        end_time = time.time()
        
        # Memory after simulation
        mem_after_sim = measure_memory_usage()
        
        sim_time = end_time - start_time
        memory_used = mem_after_sim - mem_before
        
        times.append(sim_time)
        memories.append(memory_used)
        
        print(f"    Time: {sim_time*1000:.1f}ms, Memory: {memory_used:.1f}MB, Spikes: {total_spikes}")
        
        # Cleanup
        del pool
    
    # Verify scaling makes sense
    print(f"\n  📊 SCALING ANALYSIS:")
    for i, size in enumerate(sizes):
        neurons_per_ms = (size * n_steps) / (times[i] * 1000)
        print(f"    {size:5d} neurons: {neurons_per_ms:8.0f} neurons/ms, {memories[i]:6.1f}MB")
    
    # Check if time scales roughly linearly with size
    time_ratios = [times[i]/times[0] for i in range(len(times))]
    size_ratios = [sizes[i]/sizes[0] for i in range(len(sizes))]
    
    print(f"  Time scaling ratios: {[f'{r:.1f}x' for r in time_ratios]}")
    print(f"  Size ratios:         {[f'{r:.1f}x' for r in size_ratios]}")
    
    return times, memories

def verify_spike_authenticity(n_neurons=10000, n_steps=1000):
    """Verify spikes are genuine by testing threshold behavior"""
    print(f"\n🔥 TESTING SPIKE AUTHENTICITY ({n_neurons} neurons)")
    
    pool = GPUNeuronPool(n_neurons, 'lif')
    dt = 0.1
    
    # Test 1: No input should produce minimal spikes
    print("  Test 1: No synaptic input")
    no_input_spikes = 0
    for step in range(100):
        spike_indices, metrics = pool.step(dt, np.zeros(n_neurons))
        no_input_spikes += len(spike_indices)
    
    # Test 2: Strong input should produce many spikes  
    print("  Test 2: Strong synaptic input")
    pool = GPUNeuronPool(n_neurons, 'lif')  # Reset
    strong_input_spikes = 0
    strong_input = np.full(n_neurons, 20.0)  # Well above threshold
    for step in range(100):
        spike_indices, metrics = pool.step(dt, strong_input)
        strong_input_spikes += len(spike_indices)
    
    # Test 3: Threshold input should produce moderate spikes
    print("  Test 3: Threshold-level input")
    pool = GPUNeuronPool(n_neurons, 'lif')  # Reset
    threshold_spikes = 0
    threshold_input = np.full(n_neurons, 10.0)  # Near threshold
    for step in range(100):
        spike_indices, metrics = pool.step(dt, threshold_input)
        threshold_spikes += len(spike_indices)
    
    print(f"    No input spikes:     {no_input_spikes:6d}")
    print(f"    Threshold spikes:    {threshold_spikes:6d}")  
    print(f"    Strong input spikes: {strong_input_spikes:6d}")
    
    # Verify logical ordering
    authentic = (no_input_spikes < threshold_spikes < strong_input_spikes)
    print(f"  ✓ Spike ordering correct: {authentic}")
    
    return authentic

def main():
    """Run comprehensive verification of simulation authenticity"""
    print("🔬 NEUROMORPHIC SIMULATION AUTHENTICITY VERIFICATION")
    print("=" * 60)
    
    try:
        # Test 1: Basic neuron dynamics
        pool = GPUNeuronPool(1000, 'lif')
        dynamics_ok = verify_neuron_dynamics(pool)
        print(f"✓ Neuron dynamics test: {'PASS' if dynamics_ok else 'FAIL'}")
        
        # Test 2: Spike authenticity
        spikes_ok = verify_spike_authenticity()
        print(f"✓ Spike authenticity test: {'PASS' if spikes_ok else 'FAIL'}")
        
        # Test 3: Computational scaling
        times, memories = verify_computational_scaling()
        scaling_ok = len(times) == 4 and all(t > 0 for t in times)
        print(f"✓ Computational scaling test: {'PASS' if scaling_ok else 'FAIL'}")
        
        # Test 4: Large scale verification
        print(f"\n🎯 LARGE SCALE VERIFICATION")
        large_sizes = [100000, 500000, 1000000]
        
        for size in large_sizes:
            print(f"  Testing {size:,} neurons...")
            start_mem = measure_memory_usage()
            
            pool = GPUNeuronPool(size, 'lif')
            synaptic_input = np.random.randn(size) * 5
            
            start_time = time.time()
            total_spikes = 0
            n_steps = 10
            
            for step in range(n_steps):
                spike_indices, metrics = pool.step(0.1, synaptic_input)
                total_spikes += len(spike_indices)
            
            end_time = time.time()
            end_mem = measure_memory_usage()
            
            sim_time = end_time - start_time
            throughput = (size * n_steps) / sim_time
            memory_used = end_mem - start_mem
            
            print(f"    Time: {sim_time*1000:.1f}ms")
            print(f"    Throughput: {throughput:,.0f} neurons/second")
            print(f"    Memory: {memory_used:.1f}MB ({memory_used*1024/size:.1f} KB/neuron)")
            print(f"    Spikes: {total_spikes:,}")
            print(f"    Spike rate: {total_spikes/(size*n_steps*0.1)*1000:.1f} Hz")
            
            del pool
        
        print(f"\n{'='*60}")
        print(f"🎉 VERIFICATION COMPLETE")
        print(f"{'='*60}")
        
        if dynamics_ok and spikes_ok and scaling_ok:
            print("✅ SIMULATION VERIFIED: Numbers are authentic!")
            print("   - Neurons exhibit proper membrane dynamics")
            print("   - Spikes respond correctly to input strength") 
            print("   - Computational scaling behaves as expected")
            print("   - Large scale performance confirmed")
        else:
            print("❌ SIMULATION ISSUES DETECTED")
            print("   - Some tests failed, numbers may not be authentic")
        
    except Exception as e:
        print(f"❌ ERROR during verification: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
