#!/usr/bin/env python3
"""
EXTREME SCALE STRESS TEST - Push RTX 3060 to absolute limits
Monitor GPU usage while we find the breaking point!
"""

import numpy as np
import time
import psutil
import os
import gc
from core.gpu_neurons import GPUNeuronPool

def measure_memory_usage():
    """Get current system memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def extreme_scale_stress_test():
    """Push the GPU to its absolute limits"""
    print("🔥 EXTREME SCALE STRESS TEST - RTX 3060 LIMIT FINDER")
    print("=" * 70)
    print("🚨 WARNING: This will push your GPU to maximum capacity!")
    print("   Monitor your GPU temperature and memory usage!")
    print("=" * 70)
    
    # Progressive scale test - start big and go bigger
    test_sizes = [
        10_000_000,    # 10M - warm up
        25_000_000,    # 25M - getting serious
        50_000_000,    # 50M - proven working
        75_000_000,    # 75M - pushing limits
        100_000_000,   # 100M - approaching human cortex scale
        125_000_000,   # 125M - massive scale
        150_000_000,   # 150M - extreme scale
        200_000_000,   # 200M - insane scale
        250_000_000,   # 250M - breaking point?
        300_000_000,   # 300M - theoretical limit
    ]
    
    successful_runs = []
    max_achieved = 0
    
    for size in test_sizes:
        print(f"\n🧠 ATTEMPTING {size:,} NEURONS...")
        print(f"   Expected memory: ~{size * 60 / 1024 / 1024:.1f}MB GPU + {size * 4 / 1024 / 1024:.1f}MB system")
        
        try:
            # Memory before
            sys_mem_before = measure_memory_usage()
            
            # Create the monster network
            print(f"   🔧 Creating neuron pool...")
            start_creation = time.time()
            pool = GPUNeuronPool(size, 'lif')
            end_creation = time.time()
            
            creation_time = end_creation - start_creation
            sys_mem_after_creation = measure_memory_usage()
            
            print(f"   ✅ Creation successful! Time: {creation_time:.1f}s")
            print(f"   📊 System memory used: {sys_mem_after_creation - sys_mem_before:.1f}MB")
            
            # Test simulation performance
            print(f"   🏃 Testing simulation performance...")
            
            # Generate realistic input
            synaptic_input = np.random.randn(size) * 3
            
            # Run a few steps to measure performance
            step_times = []
            total_spikes = 0
            
            for step in range(3):  # Just 3 steps for stress test
                print(f"     Step {step+1}/3...", end=" ", flush=True)
                
                step_start = time.time()
                spike_indices, metrics = pool.step(0.1, synaptic_input)
                step_end = time.time()
                
                step_time = step_end - step_start
                step_times.append(step_time)
                num_spikes = len(spike_indices)
                total_spikes += num_spikes
                
                print(f"{step_time*1000:.1f}ms, {num_spikes:,} spikes")
            
            # Calculate performance metrics
            avg_step_time = np.mean(step_times)
            throughput = size / avg_step_time
            sys_mem_final = measure_memory_usage()
            total_sys_memory = sys_mem_final - sys_mem_before
            
            # Success!
            print(f"   🎉 SUCCESS! Network operational!")
            print(f"     Average step time:  {avg_step_time*1000:.1f}ms")
            print(f"     Throughput:         {throughput:,.0f} neurons/second")
            print(f"     Total spikes:       {total_spikes:,}")
            print(f"     System memory:      {total_sys_memory:.1f}MB")
            print(f"     Memory per neuron:  {total_sys_memory*1024/size:.2f} KB")
            
            # Performance classification
            if throughput > 1_000_000_000:  # 1B+ neurons/sec
                performance = "🚀 INCREDIBLE"
            elif throughput > 500_000_000:  # 500M+ neurons/sec
                performance = "⚡ EXCELLENT"
            elif throughput > 100_000_000:  # 100M+ neurons/sec
                performance = "✨ VERY GOOD"
            else:
                performance = "📊 FUNCTIONAL"
            
            print(f"     Performance:        {performance}")
            
            # Record success
            successful_runs.append({
                'size': size,
                'step_time': avg_step_time,
                'throughput': throughput,
                'memory': total_sys_memory,
                'spikes': total_spikes,
                'creation_time': creation_time
            })
            
            max_achieved = size
            
            # Cleanup before next test
            del pool
            gc.collect()  # Force garbage collection
            time.sleep(2)  # Let GPU cool down
            
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
            print(f"   🏁 MAXIMUM SCALE REACHED: {max_achieved:,} neurons")
            break
    
    # Summary of achievements
    print(f"\n{'='*70}")
    print(f"🏆 EXTREME SCALE TEST RESULTS")
    print(f"{'='*70}")
    
    if successful_runs:
        print(f"🎯 MAXIMUM ACHIEVED: {max_achieved:,} neurons")
        print(f"📊 SUCCESSFUL RUNS:")
        
        for run in successful_runs:
            size = run['size']
            throughput = run['throughput']
            step_time = run['step_time'] * 1000
            memory = run['memory']
            
            print(f"   {size:>11,} neurons: {step_time:6.1f}ms, {throughput:>10,.0f} n/s, {memory:5.1f}MB")
        
        # Find peak performance
        best_throughput = max(run['throughput'] for run in successful_runs)
        best_run = next(run for run in successful_runs if run['throughput'] == best_throughput)
        
        print(f"\n🏆 PEAK PERFORMANCE:")
        print(f"   Network size:    {best_run['size']:,} neurons")
        print(f"   Throughput:      {best_run['throughput']:,.0f} neurons/second")
        print(f"   Step time:       {best_run['step_time']*1000:.1f}ms")
        print(f"   Memory usage:    {best_run['memory']:.1f}MB")
        
        # Scale comparison
        human_cortex = 16_000_000_000  # 16B neurons in human cortex
        human_brain = 86_000_000_000   # 86B neurons in human brain
        
        cortex_percent = (max_achieved / human_cortex) * 100
        brain_percent = (max_achieved / human_brain) * 100
        
        print(f"\n🧠 BIOLOGICAL SCALE COMPARISON:")
        print(f"   vs Human Cortex:  {cortex_percent:.2f}% ({max_achieved:,}/{human_cortex:,})")
        print(f"   vs Human Brain:   {brain_percent:.3f}% ({max_achieved:,}/{human_brain:,})")
        
        # Hardware achievement
        print(f"\n💻 HARDWARE ACHIEVEMENT:")
        print(f"   GPU: NVIDIA RTX 3060 (8GB)")
        print(f"   Status: {'🔥 MAXIMUM CAPACITY REACHED' if len(successful_runs) < len(test_sizes) else '🚀 ALL TESTS PASSED'}")
        
    else:
        print(f"❌ NO SUCCESSFUL RUNS - Check GPU memory!")

if __name__ == "__main__":
    print("🔥 Starting in 3 seconds... Monitor your GPU now!")
    time.sleep(3)
    extreme_scale_stress_test()
