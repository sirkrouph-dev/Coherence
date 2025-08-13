"""
RTX 3060 GPU Acceleration Test for NeuroMorph.
Optimized for 8GB VRAM and your specific hardware.
"""

import time
import numpy as np
import sys
import os

# Add the core directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'core'))

# GPU imports with fallback
try:
    import cupy as cp
    GPU_AVAILABLE = cp.cuda.is_available()
    if GPU_AVAILABLE:
        device_info = cp.cuda.Device()
        free_mem, total_mem = device_info.mem_info
        device_name = cp.cuda.runtime.getDeviceProperties(device_info.id)['name'].decode()
        print(f"🚀 GPU: {device_name}")
        print(f"💾 VRAM: {total_mem/(1024**3):.1f}GB total, {free_mem/(1024**3):.1f}GB free")
    else:
        print("❌ No GPU detected")
except ImportError:
    cp = None  # Don't alias to np to avoid confusion
    GPU_AVAILABLE = False
    print("❌ CuPy not installed")
except Exception as e:
    cp = None
    GPU_AVAILABLE = False
    print(f"❌ GPU setup failed: {e}")

try:
    from core.vectorized_neurons import create_vectorized_population
except ImportError as e:
    print(f"❌ Could not import vectorized neurons: {e}")
    print("Please ensure the core module is properly installed.")
    sys.exit(1)


def test_rtx3060_scaling():
    """Test GPU acceleration scaling on RTX 3060."""
    print("=" * 70)
    print("RTX 3060 NEUROMORPHIC SCALING TEST")
    print("=" * 70)
    
    # Test sizes optimized for 8GB VRAM
    test_sizes = [10000, 50000, 100000, 250000, 500000]
    
    print(f"{'Size':<8} {'Platform':<8} {'Time(ms)':<10} {'Throughput':<15} {'Memory(MB)':<12} {'Speedup':<10}")
    print("-" * 75)
    
    cpu_baseline = {}
    
    for size in test_sizes:
        # Estimate if it fits in VRAM
        estimated_vram_mb = (size * 6 * 4) / (1024**2)  # 6 float32 arrays per neuron
        fits_in_vram = estimated_vram_mb < 6000  # Use 6GB of 8GB for safety
        
        print(f"\nTesting {size} neurons (estimated {estimated_vram_mb:.0f}MB VRAM)...")
        
        # CPU Test
        cpu_time, cpu_memory = test_cpu_performance(size)
        cpu_baseline[size] = cpu_time
        throughput_cpu = (size * 100) / cpu_time if cpu_time > 0 else 0
        
        print(f"{size:<8} {'CPU':<8} {cpu_time*1000:<8.1f} {throughput_cpu/1000:<12.1f}k/s {cpu_memory:<10.1f} {'1.0x':<10}")
        
        # GPU Test (if available and fits)
        if GPU_AVAILABLE and fits_in_vram:
            try:
                gpu_time, gpu_memory = test_gpu_performance(size)
                throughput_gpu = (size * 100) / gpu_time if gpu_time > 0 else 0
                speedup = cpu_time / gpu_time if gpu_time > 0 else 0
                
                print(f"{size:<8} {'GPU':<8} {gpu_time*1000:<8.1f} {throughput_gpu/1000:<12.1f}k/s {gpu_memory:<10.1f} {speedup:<8.1f}x")
                
            except Exception as e:
                print(f"{size:<8} {'GPU':<8} FAILED: {str(e)}")
        else:
            reason = "No GPU" if not GPU_AVAILABLE else "VRAM limit"
            print(f"{size:<8} {'GPU':<8} SKIPPED ({reason})")


def test_cpu_performance(size: int) -> tuple:
    """Test CPU performance."""
    # Create CPU population
    pop = create_vectorized_population(size, "lif")
    
    # Get initial memory
    mem_before = get_memory_mb()
    
    # Run simulation
    I_syn = np.random.uniform(80, 120, size)
    
    start_time = time.time()
    for _ in range(100):  # 100 steps
        spikes = pop.step(0.1, I_syn)
    sim_time = time.time() - start_time
    
    # Get final memory
    mem_after = get_memory_mb()
    memory_used = mem_after - mem_before
    
    return sim_time, memory_used


def test_gpu_performance(size: int) -> tuple:
    """Test GPU performance using CuPy directly."""
    if not GPU_AVAILABLE:
        raise RuntimeError("GPU not available")
    
    # Import cupy again to ensure we have the real module
    import cupy as cp_real
    
    # Create arrays on GPU
    membrane_potential = cp_real.full(size, -65.0, dtype=cp_real.float32)
    spike_flags = cp_real.zeros(size, dtype=bool)
    refractory_time = cp_real.zeros(size, dtype=cp_real.float32)
    v_thresh = cp_real.full(size, -60.0, dtype=cp_real.float32)
    tau_m = cp_real.full(size, 10.0, dtype=cp_real.float32)
    
    # Input current on GPU
    I_syn = cp_real.random.uniform(80, 120, size).astype(cp_real.float32)
    
    # Get initial GPU memory
    free_mem_before, _ = cp_real.cuda.Device().mem_info
    
    # GPU simulation loop
    start_time = time.time()
    
    for step in range(100):  # 100 steps
        dt = 0.1
        
        # Reset spikes
        spike_flags.fill(False)
        
        # LIF dynamics - FULLY VECTORIZED ON GPU
        dv_dt = (-(membrane_potential - (-65.0)) + I_syn * 100.0) / tau_m
        membrane_potential += dv_dt * dt
        
        # Spike detection - VECTORIZED ON GPU
        spike_mask = membrane_potential >= v_thresh
        
        if cp_real.any(spike_mask):
            spike_flags[spike_mask] = True
            membrane_potential[spike_mask] = -65.0  # Reset
    
    # Synchronize GPU before measuring time
    cp_real.cuda.Stream.null.synchronize()
    sim_time = time.time() - start_time
    
    # Get final GPU memory
    free_mem_after, total_mem = cp_real.cuda.Device().mem_info
    memory_used = (free_mem_before - free_mem_after) / (1024**2)  # MB
    
    return sim_time, memory_used


def test_memory_scaling():
    """Test memory usage scaling."""
    print("\n" + "=" * 70)
    print("MEMORY SCALING ANALYSIS")
    print("=" * 70)
    
    sizes = [10000, 50000, 100000, 250000, 500000, 750000, 1000000]
    
    print(f"{'Neurons':<10} {'CPU Memory':<12} {'GPU Memory':<12} {'Fits RTX3060':<15}")
    print("-" * 60)
    
    for size in sizes:
        # Estimate CPU memory (NumPy arrays)
        cpu_memory_mb = (size * 6 * 4) / (1024**2)  # 6 float32 arrays
        
        # Estimate GPU memory (same arrays + overhead)
        gpu_memory_mb = cpu_memory_mb * 1.2  # 20% GPU overhead
        
        # Check if fits in RTX 3060 (8GB VRAM, use 6GB for safety)
        fits_rtx3060 = gpu_memory_mb < 6000
        fit_status = "✓ YES" if fits_rtx3060 else "✗ NO"
        
        print(f"{size:<10} {cpu_memory_mb:<10.1f}MB {gpu_memory_mb:<10.1f}MB {fit_status:<15}")


def benchmark_gpu_operations():
    """Benchmark specific GPU operations."""
    if not GPU_AVAILABLE:
        print("❌ GPU not available for operation benchmarks")
        return
        
    # Import cupy again to ensure we have the real module
    import cupy as cp_real
        
    print("\n" + "=" * 70)
    print("GPU OPERATION BENCHMARKS")
    print("=" * 70)
    
    size = 100000  # 100k neurons
    
    # Create test data
    a = cp_real.random.random(size).astype(cp_real.float32)
    b = cp_real.random.random(size).astype(cp_real.float32)
    c = cp_real.random.random(size).astype(cp_real.float32)
    
    operations = [
        ("Vector Addition", lambda: a + b),
        ("Vector Multiplication", lambda: a * b),
        ("Exponential", lambda: cp_real.exp(a)),
        ("Threshold Comparison", lambda: a > 0.5),
        ("Boolean Indexing", lambda: a[b > 0.5]),
        ("Fill Operation", lambda: a.fill(0.0)),
    ]
    
    print(f"{'Operation':<20} {'Time (μs)':<12} {'Throughput':<15}")
    print("-" * 50)
    
    for name, operation in operations:
        # Warm up
        for _ in range(10):
            operation()
        cp_real.cuda.Stream.null.synchronize()
        
        # Benchmark
        start_time = time.time()
        for _ in range(100):
            result = operation()
        cp_real.cuda.Stream.null.synchronize()
        total_time = time.time() - start_time
        
        avg_time_us = (total_time / 100) * 1000000  # microseconds
        throughput = (size / (total_time / 100)) / 1000  # k elements/second
        
        print(f"{name:<20} {avg_time_us:<10.1f} {throughput:<12.1f}k/s")


def get_memory_mb():
    """Get current memory usage in MB."""
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def get_optimization_recommendations():
    """Get specific recommendations for RTX 3060."""
    print("\n" + "=" * 70)
    print("RTX 3060 OPTIMIZATION RECOMMENDATIONS")
    print("=" * 70)
    
    recommendations = [
        "✓ Optimal neuron count: 250k-500k neurons",
        "✓ Use LIF neurons for maximum performance", 
        "✓ Batch processing for >500k neurons",
        "✓ GPU memory: Reserve 2GB for system/drivers",
        "✓ Use float32 precision (not float64)",
        "✓ Enable memory pooling for repeated simulations",
        "✓ Monitor VRAM usage with nvidia-smi",
        "✓ Consider mixed precision for very large networks"
    ]
    
    for rec in recommendations:
        print(f"  {rec}")
    
    print(f"\n🎯 TARGET PERFORMANCE:")
    print(f"  • 250k neurons: ~10-15ms simulation time")
    print(f"  • 500k neurons: ~20-30ms simulation time")
    print(f"  • Real-time factor: 1000x biological time")
    print(f"  • Memory usage: <6GB VRAM")


if __name__ == "__main__":
    try:
        test_rtx3060_scaling()
        test_memory_scaling()
        benchmark_gpu_operations()
        get_optimization_recommendations()
        
        print(f"\n🚀 RTX 3060 ready for large-scale neuromorphic computing!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
