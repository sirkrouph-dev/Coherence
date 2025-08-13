#!/usr/bin/env python3
"""
Advanced bottleneck analysis - since thermal isn't the issue,
let's find the real performance limiters
"""

import numpy as np
import time
import psutil
import os
from core.gpu_neurons import GPUNeuronPool

def analyze_memory_patterns():
    """Analyze memory allocation patterns to find bottlenecks"""
    print("🔍 MEMORY PATTERN ANALYSIS")
    print("=" * 50)
    
    sizes = [10_000_000, 50_000_000, 100_000_000, 150_000_000, 200_000_000]
    
    for size in sizes:
        print(f"\n📊 Analyzing {size:,} neurons:")
        
        try:
            # Memory before
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024
            
            start_time = time.time()
            pool = GPUNeuronPool(size, 'lif')
            creation_time = time.time() - start_time
            
            mem_after = process.memory_info().rss / 1024 / 1024
            
            print(f"  Creation time: {creation_time*1000:.1f}ms")
            print(f"  Memory delta: {mem_after - mem_before:.1f}MB")
            print(f"  Batch size: {pool.batch_size:,}")
            
            # Test memory access patterns
            synaptic_input = np.random.randn(size) * 3
            
            # Single step timing
            step_times = []
            for i in range(5):
                start = time.time()
                spike_indices, metrics = pool.step(0.1, synaptic_input)
                end = time.time()
                step_times.append((end - start) * 1000)
                print(f"  Step {i+1}: {step_times[-1]:.1f}ms, {len(spike_indices):,} spikes")
            
            avg_time = np.mean(step_times)
            std_time = np.std(step_times)
            
            print(f"  Average: {avg_time:.1f}ms ± {std_time:.1f}ms")
            
            # Memory access efficiency
            neurons_per_ms = size / avg_time * 1000
            print(f"  Efficiency: {neurons_per_ms:,.0f} neurons/second")
            
            # Check for batch processing overhead
            if size > 10_000_000:  # Uses batching
                batch_overhead = avg_time / (size / pool.batch_size)
                print(f"  Batch overhead: {batch_overhead:.1f}ms per batch")
            
            del pool
            
        except Exception as e:
            print(f"  ❌ Error: {e}")

def analyze_batch_efficiency():
    """Test different batch sizes to find optimal configuration"""
    print(f"\n🎯 BATCH SIZE OPTIMIZATION")
    print("=" * 40)
    
    size = 100_000_000  # 100M neurons
    batch_sizes = [100_000, 500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000]
    
    for batch_size in batch_sizes:
        print(f"\n  Testing batch size: {batch_size:,}")
        
        try:
            # Simulate batch processing timing
            num_batches = size // batch_size
            print(f"    Number of batches: {num_batches}")
            
            # Create smaller test
            test_size = min(batch_size, 1_000_000)
            pool = GPUNeuronPool(test_size, 'lif')
            
            synaptic_input = np.random.randn(test_size) * 3
            
            # Time single batch
            start = time.time()
            spike_indices, metrics = pool.step(0.1, synaptic_input)
            batch_time = (time.time() - start) * 1000
            
            # Estimate total time
            estimated_total = batch_time * num_batches
            throughput = size / (estimated_total / 1000)
            
            print(f"    Batch time: {batch_time:.1f}ms")
            print(f"    Estimated total: {estimated_total:.1f}ms")
            print(f"    Estimated throughput: {throughput:,.0f} neurons/sec")
            
            del pool
            
        except Exception as e:
            print(f"    ❌ Error: {e}")

def analyze_memory_bandwidth():
    """Test memory bandwidth limitations"""
    print(f"\n💾 MEMORY BANDWIDTH ANALYSIS")
    print("=" * 40)
    
    # Test different data access patterns
    sizes = [1_000_000, 10_000_000, 50_000_000, 100_000_000]
    
    for size in sizes:
        print(f"\n  Testing {size:,} neurons:")
        
        try:
            pool = GPUNeuronPool(size, 'lif')
            
            # Test different input patterns
            patterns = {
                "zeros": np.zeros(size),
                "random": np.random.randn(size) * 3,
                "constant": np.full(size, 10.0),
                "sparse": np.zeros(size)
            }
            patterns["sparse"][:size//1000] = 15.0  # 0.1% active
            
            for pattern_name, synaptic_input in patterns.items():
                times = []
                for _ in range(3):
                    start = time.time()
                    spike_indices, metrics = pool.step(0.1, synaptic_input)
                    times.append((time.time() - start) * 1000)
                
                avg_time = np.mean(times)
                bandwidth = (size * 4 * 2) / (avg_time / 1000) / 1e9  # GB/s (read + write)
                
                print(f"    {pattern_name:8s}: {avg_time:6.1f}ms, {bandwidth:5.1f} GB/s")
            
            del pool
            
        except Exception as e:
            print(f"    ❌ Error: {e}")

def main():
    """Run comprehensive bottleneck analysis"""
    print("🔬 ADVANCED BOTTLENECK ANALYSIS")
    print("Since thermal isn't the issue (50°C is excellent!),")
    print("let's find the real performance limiters...")
    print("=" * 60)
    
    analyze_memory_patterns()
    analyze_batch_efficiency()
    analyze_memory_bandwidth()
    
    print(f"\n{'='*60}")
    print(f"📋 ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"🌡️  Thermal: NOT the bottleneck (50°C is excellent)")
    print(f"🔍  Real limiters: Check results above for:")
    print(f"   - Memory bandwidth saturation")
    print(f"   - Batch processing overhead")
    print(f"   - GPU memory allocation patterns")
    print(f"   - CUDA kernel launch overhead")

if __name__ == "__main__":
    main()
