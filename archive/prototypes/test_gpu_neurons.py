"""
Quick test of GPU-accelerated neuron scaling
"""

import sys
import time
import numpy as np

from core.gpu_neurons import GPUNeuronPool, MultiGPUNeuronSystem

def test_basic_scaling():
    """Test basic GPU neuron scaling."""
    print("\n" + "="*60)
    print("GPU NEURON SCALING TEST")
    print("="*60)
    
    # Test different scales
    test_scales = [1000, 5000, 10000, 50000, 100000]
    
    for num_neurons in test_scales:
        print(f"\n📊 Testing {num_neurons:,} neurons...")
        
        try:
            # Create GPU neuron pool
            pool = GPUNeuronPool(
                num_neurons=num_neurons,
                neuron_type="adex",
                use_gpu=True,
                batch_size=min(10000, num_neurons)
            )
            
            # Run short simulation
            total_spikes = 0
            start_time = time.time()
            
            for step in range(100):  # 100 steps
                # Generate input
                I_syn = np.random.randn(num_neurons) * 10
                
                # Step simulation
                spikes, metrics = pool.step(0.1, I_syn)
                total_spikes += len(spikes)
            
            elapsed = time.time() - start_time
            
            # Get statistics
            stats = pool.get_spike_statistics()
            
            print(f"  ✅ Success!")
            print(f"     Time: {elapsed:.3f}s")
            print(f"     Throughput: {num_neurons * 100 / elapsed:,.0f} neurons/sec")
            print(f"     Total spikes: {total_spikes:,}")
            print(f"     GPU memory: {metrics.get('gpu_memory_used_mb', 0):.1f} MB")
            
            # Cleanup
            pool.clear_gpu_memory()
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")

def test_million_neurons():
    """Test million neuron simulation."""
    print("\n" + "="*60)
    print("MILLION NEURON TEST")
    print("="*60)
    
    try:
        print("\n🚀 Creating 1 million neuron system...")
        multi_system = MultiGPUNeuronSystem(
            total_neurons=1000000,
            neurons_per_gpu=100000
        )
        
        print("\n⚡ Running short simulation (10ms)...")
        results = multi_system.simulate(duration=10.0, dt=0.1)
        
        print("\n📊 Results:")
        print(f"  Total neurons: {results['total_neurons']:,}")
        print(f"  Total spikes: {results['total_spikes']:,}")
        print(f"  Simulation time: {results['simulation_time']:.2f}s")
        print(f"  Throughput: {results['neurons_per_second']:,.0f} neurons/sec")
        
        if "total_gpu_memory_mb" in results:
            print(f"  GPU memory: {results['total_gpu_memory_mb']:.1f} MB")
        
        # Cleanup
        multi_system.cleanup()
        
    except Exception as e:
        print(f"\n❌ Million neuron test failed: {e}")

def main():
    """Main test function."""
    print("\n🧠 GPU-ACCELERATED NEURON SYSTEM TEST")
    
    # Check GPU availability
    try:
        import cupy as cp
        print(f"✓ CuPy available - GPU acceleration enabled")
        
        # Get GPU info
        device = cp.cuda.Device()
        print(f"  Device: Compute capability {device.compute_capability}")
        
    except ImportError:
        print("⚠ CuPy not available - running in CPU mode")
    
    # Run tests
    test_basic_scaling()
    test_million_neurons()
    
    print("\n✅ Testing complete!")

if __name__ == "__main__":
    main()
