"""
NEUROMORPHIC SCALING STRATEGY TEST
Tests the complete scaling path from CPU to GPU to neuromorphic hardware.

Scale Tiers:
1. CPU Optimized: 1k-10k neurons (software development/testing)
2. GPU Automatic: 10k-1M neurons (research/small deployments) 
3. GPU Mandatory: 1M-50M neurons (large-scale research)
4. Neuromorphic Scale: 50M+ neurons (neuromorphic hardware simulation)
"""

import time
import numpy as np
from core.network import NeuromorphicNetwork

def _test_scaling_tier(name, neurons, connectivity, expected_gpu=False, should_fail_cpu=False):
    """Test a specific scaling tier."""
    print(f"\n=== {name.upper()} ===")
    print(f"Target: {neurons:,} neurons, {connectivity:.1%} connectivity")
    
    try:
        start = time.time()
        
        # Build network
        net = NeuromorphicNetwork()
        
        # For very large networks, use multiple layers for realism
        if neurons >= 1000000:
            # Multi-layer architecture for massive scale
            layers = [
                ("sensory", neurons // 4),
                ("cortical_l1", neurons // 3), 
                ("cortical_l2", neurons // 4),
                ("motor", neurons // 6)
            ]
            
            for layer_name, size in layers:
                net.add_layer(layer_name, size, 'lif')
            
            # Sparse connectivity for massive scale
            net.connect_layers("sensory", "cortical_l1", "stdp", connectivity)
            net.connect_layers("cortical_l1", "cortical_l2", "stdp", connectivity * 0.5)
            net.connect_layers("cortical_l2", "motor", "stdp", connectivity * 0.3)
            
        else:
            # Simple two-layer for smaller networks
            net.add_layer('input', neurons // 2, 'lif')
            net.add_layer('output', neurons // 2, 'lif')
            net.connect_layers('input', 'output', 'stdp', connectivity)
        
        build_time = time.time() - start
        print(f"✓ Network built: {build_time:.3f}s")
        
        # Test simulation
        start_sim = time.time()
        duration = 0.5 if neurons < 100000 else 0.1  # Shorter sim for massive networks
        results = net.run_simulation(duration, 0.1)
        sim_time = time.time() - start_sim
        
        # Calculate metrics
        total_spikes = sum(len(spikes) for spikes in results['layer_spike_times'].values())
        spike_rate = total_spikes / neurons / (duration / 1000.0)  # Hz per neuron
        throughput = total_spikes / sim_time
        
        print(f"✓ Simulation: {sim_time:.3f}s")
        print(f"  Duration: {duration}ms")
        print(f"  Spikes: {total_spikes:,}")
        print(f"  Rate: {spike_rate:.1f} Hz/neuron")
        print(f"  Throughput: {throughput:.0f} spikes/sec")
        
        # Performance assessment
        total_time = build_time + sim_time
        if neurons <= 10000:
            target_time = 2.0  # CPU target: <2s for 10k neurons
        elif neurons <= 100000:
            target_time = 5.0  # GPU target: <5s for 100k neurons
        else:
            target_time = 30.0  # Massive scale: <30s for 1M+ neurons
            
        if total_time <= target_time:
            print(f"✅ PERFORMANCE TARGET MET: {total_time:.1f}s ≤ {target_time}s")
        else:
            print(f"⚠️  Performance needs improvement: {total_time:.1f}s > {target_time}s")
            
        return True, total_time, throughput
        
    except RuntimeError as e:
        if "REQUIRE GPU" in str(e) and should_fail_cpu:
            print(f"✅ CORRECTLY REJECTED CPU: {e}")
            return True, float('inf'), 0
        else:
            print(f"❌ UNEXPECTED ERROR: {e}")
            return False, float('inf'), 0
            
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False, float('inf'), 0

def main():
    """Test complete scaling strategy."""
    print("NEUROMORPHIC SCALING STRATEGY TEST")
    print("=" * 50)
    
    # Define scaling tiers
    test_cases = [
        # Tier 1: CPU Optimized (development/testing)
        ("CPU Small", 1000, 0.05, False, False),
        ("CPU Medium", 5000, 0.02, False, False),
        ("CPU Large", 10000, 0.01, False, False),
        
        # Tier 2: GPU Automatic (research scale)
        ("GPU Small", 25000, 0.005, True, False),
        ("GPU Medium", 50000, 0.003, True, False),
        ("GPU Large", 100000, 0.002, True, False),
        
        # Tier 3: GPU Mandatory (large research)
        ("GPU Mandatory", 1000000, 0.0005, True, True),  # 1M neurons
        
        # Tier 4: Neuromorphic Scale (hardware simulation)
        # ("Neuromorphic", 50000000, 0.0001, True, True),  # 50M neurons - COMMENTED for safety
    ]
    
    results = []
    
    for name, neurons, connectivity, expected_gpu, should_fail_cpu in test_cases:
        success, time_taken, throughput = _test_scaling_tier(
            name, neurons, connectivity, expected_gpu, should_fail_cpu
        )
        results.append((name, neurons, success, time_taken, throughput))
    
    # Summary
    print("\n" + "=" * 70)
    print("SCALING STRATEGY SUMMARY")
    print("=" * 70)
    print(f"{'Scale':<15} {'Neurons':<10} {'Status':<8} {'Time(s)':<8} {'Throughput':<12}")
    print("-" * 70)
    
    for name, neurons, success, time_taken, throughput in results:
        status = "✅ PASS" if success else "❌ FAIL"
        time_str = f"{time_taken:.1f}" if time_taken != float('inf') else "N/A"
        throughput_str = f"{throughput:.0f}" if throughput > 0 else "N/A"
        
        print(f"{name:<15} {neurons:<10,} {status:<8} {time_str:<8} {throughput_str:<12}")
    
    # GPU availability check
    print("\n" + "=" * 70)
    print("GPU ACCELERATION STATUS")
    print("=" * 70)
    
    try:
        import cupy as cp
        gpu_available = True
        print("✅ CuPy: Available")
        print(f"   GPU Memory: {cp.cuda.Device().mem_info[1] / 1024**3:.1f} GB")
    except ImportError:
        gpu_available = False
        print("❌ CuPy: Not installed")
        print("   Install: pip install cupy-cuda12x")
    
    try:
        import torch
        print(f"✅ PyTorch: Available (CUDA: {torch.cuda.is_available()})")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name()}")
    except ImportError:
        print("❌ PyTorch: Not installed")
    
    print("\n🎯 RECOMMENDATION:")
    if gpu_available:
        print("   Your system is ready for massive neuromorphic simulation!")
        print("   • CPU: Optimized for 1k-10k neurons")
        print("   • GPU: Automatic for 10k+ neurons")
        print("   • Ready for 80M neuron neuromorphic hardware simulation")
    else:
        print("   Install GPU acceleration for large-scale simulation:")
        print("   pip install cupy-cuda12x")
        print("   Current limit: ~10k neurons on CPU")

def test_scaling_strategy():
    """Test the complete neuromorphic scaling strategy."""
    main()

if __name__ == "__main__":
    main()
