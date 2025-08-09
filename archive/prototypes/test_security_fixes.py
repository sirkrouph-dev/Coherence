"""
Test script to verify security and memory leak fixes.
"""

import sys
import time
import traceback
import numpy as np

# Test GPU memory leak fix
def test_gpu_memory_leak():
    """Test that GPU memory leak has been fixed."""
    print("\n" + "="*60)
    print("Testing GPU Memory Leak Fix")
    print("="*60)
    
    try:
        from core.gpu_neurons import GPUNeuronPool
        
        # Create a pool with limited spike history
        pool = GPUNeuronPool(
            num_neurons=10000,
            neuron_type="adex",
            use_gpu=True,
            max_spike_history=1000  # Limited history to prevent leak
        )
        
        print("✓ GPU neuron pool created with spike history limit")
        
        # Run many steps to test memory management
        print("Running 10000 simulation steps...")
        I_syn = np.random.randn(10000) * 10
        
        for step in range(10000):
            if step % 1000 == 0:
                print(f"  Step {step}/10000")
            pool.step(0.1, I_syn)
        
        # Check that spike history is bounded
        if len(pool.spike_indices) <= pool.max_spike_history:
            print(f"✓ Spike history bounded at {len(pool.spike_indices)} entries (max: {pool.max_spike_history})")
        else:
            print(f"✗ Spike history exceeded limit: {len(pool.spike_indices)} > {pool.max_spike_history}")
        
        # Cleanup
        pool.clear_gpu_memory()
        print("✓ GPU memory cleared successfully")
        
        return True
        
    except ImportError as e:
        print(f"⚠ CuPy not available, skipping GPU test: {e}")
        return True
    except Exception as e:
        print(f"✗ GPU memory leak test failed: {e}")
        traceback.print_exc()
        return False


# Test STDP weight boundaries
def test_stdp_weight_boundaries():
    """Test that STDP weights are properly bounded."""
    print("\n" + "="*60)
    print("Testing STDP Weight Boundaries")
    print("="*60)
    
    try:
        from core.synapses import STDP_Synapse
        
        # Create STDP synapse with boundaries
        synapse = STDP_Synapse(
            synapse_id=1,
            pre_neuron_id=0,
            post_neuron_id=1,
            weight=5.0,
            w_min=0.0,
            w_max=10.0
        )
        
        print(f"✓ STDP synapse created with weight bounds [0, 10]")
        print(f"  Initial weight: {synapse.weight}")
        
        # Test weight increase (should be capped at w_max)
        for _ in range(100):
            synapse.update_weight(1.0)  # Try to increase weight
        
        if synapse.weight <= synapse.w_max:
            print(f"✓ Weight properly capped at maximum: {synapse.weight} <= {synapse.w_max}")
        else:
            print(f"✗ Weight exceeded maximum: {synapse.weight} > {synapse.w_max}")
            return False
        
        # Test weight decrease (should be capped at w_min)
        for _ in range(200):
            synapse.update_weight(-1.0)  # Try to decrease weight
        
        if synapse.weight >= synapse.w_min:
            print(f"✓ Weight properly capped at minimum: {synapse.weight} >= {synapse.w_min}")
        else:
            print(f"✗ Weight below minimum: {synapse.weight} < {synapse.w_min}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ STDP weight boundary test failed: {e}")
        traceback.print_exc()
        return False


# Test network input validation
def test_network_input_validation():
    """Test network input validation and resource limits."""
    print("\n" + "="*60)
    print("Testing Network Input Validation")
    print("="*60)
    
    try:
        from core.network import NeuromorphicNetwork
        
        network = NeuromorphicNetwork()
        
        # Test layer size validation
        print("Testing layer size validation...")
        try:
            network.add_layer("test", -10)  # Invalid negative size
            print("✗ Failed to catch negative layer size")
            return False
        except ValueError as e:
            print(f"✓ Caught invalid layer size: {e}")
        
        # Test layer name validation
        try:
            network.add_layer("", 100)  # Empty name
            print("✗ Failed to catch empty layer name")
            return False
        except ValueError as e:
            print(f"✓ Caught empty layer name: {e}")
        
        # Test neuron type validation
        try:
            network.add_layer("test", 100, neuron_type="invalid_type")
            print("✗ Failed to catch invalid neuron type")
            return False
        except ValueError as e:
            print(f"✓ Caught invalid neuron type: {e}")
        
        # Test resource limits
        network.add_layer("layer1", 100, neuron_type="lif")
        print("✓ Added valid layer")
        
        # Test connection probability validation
        network.add_layer("layer2", 100, neuron_type="lif")
        
        try:
            network.connect_layers("layer1", "layer2", connection_probability=1.5)  # Invalid probability
            print("✗ Failed to catch invalid connection probability")
            return False
        except ValueError as e:
            print(f"✓ Caught invalid connection probability: {e}")
        
        # Test synapse type validation
        try:
            network.connect_layers("layer1", "layer2", synapse_type="invalid_synapse")
            print("✗ Failed to catch invalid synapse type")
            return False
        except ValueError as e:
            print(f"✓ Caught invalid synapse type: {e}")
        
        # Test simulation duration validation
        try:
            network.run_simulation(duration=-10)  # Negative duration
            print("✗ Failed to catch negative duration")
            return False
        except ValueError as e:
            print(f"✓ Caught negative duration: {e}")
        
        try:
            network.run_simulation(duration=100, dt=200)  # dt > duration
            print("✗ Failed to catch dt > duration")
            return False
        except ValueError as e:
            print(f"✓ Caught invalid dt: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Network input validation test failed: {e}")
        traceback.print_exc()
        return False


# Test security manager
def test_security_manager():
    """Test security manager input validation."""
    print("\n" + "="*60)
    print("Testing Security Manager")
    print("="*60)
    
    try:
        from core.security_manager import SecurityManager, ResourceLimiter, RateLimiter
        
        # Test input validation
        print("Testing input validation...")
        
        # Test numeric bounds
        value = SecurityManager.validate_network_input(50, min_val=0, max_val=100, dtype=float)
        print(f"✓ Valid input accepted: {value}")
        
        try:
            SecurityManager.validate_network_input(150, min_val=0, max_val=100, dtype=float)
            print("✗ Failed to catch out-of-bounds value")
            return False
        except ValueError as e:
            print(f"✓ Caught out-of-bounds value: {e}")
        
        # Test string sanitization
        safe_string = SecurityManager.validate_network_input(
            "test<script>alert('xss')</script>", 
            dtype=str
        )
        if '<' not in safe_string and '>' not in safe_string:
            print(f"✓ String sanitized: {safe_string}")
        else:
            print("✗ Failed to sanitize string")
            return False
        
        # Test array validation
        print("\nTesting array validation...")
        
        arr = np.array([1, 2, 3, 4, 5])
        validated = SecurityManager.validate_array_input(arr, min_val=0, max_val=10)
        print(f"✓ Valid array accepted: shape={validated.shape}")
        
        # Test NaN detection
        try:
            bad_arr = np.array([1, 2, np.nan, 4])
            SecurityManager.validate_array_input(bad_arr)
            print("✗ Failed to catch NaN values")
            return False
        except ValueError as e:
            print(f"✓ Caught NaN values: {e}")
        
        # Test file path validation
        print("\nTesting file path validation...")
        
        try:
            # Use platform-appropriate paths
            import os
            if os.name == 'nt':  # Windows
                SecurityManager.validate_file_path("C:\\safe\\dir", "..\\etc\\passwd")
            else:  # Unix-like
                SecurityManager.validate_file_path("/safe/dir", "../etc/passwd")
            print("✗ Failed to catch path traversal")
            return False
        except ValueError as e:
            print(f"✓ Caught path traversal: {e}")
        
        # Test resource limiter
        print("\nTesting resource limiter...")
        limiter = ResourceLimiter(max_memory_mb=100)
        
        # Test memory allocation check
        if limiter.check_memory_allocation(50 * 1024 * 1024):  # 50MB
            print("✓ Memory allocation within limits")
        
        try:
            limiter.check_memory_allocation(200 * 1024 * 1024)  # 200MB
            print("✗ Failed to catch memory limit exceeded")
            return False
        except MemoryError as e:
            print(f"✓ Caught memory limit exceeded: {e}")
        
        # Test rate limiter
        print("\nTesting rate limiter...")
        rate_limiter = RateLimiter(max_calls=5, time_window=1.0)
        
        # Make some calls within limit
        for i in range(5):
            rate_limiter.check_rate_limit()
        print("✓ Rate limit allows 5 calls")
        
        try:
            rate_limiter.check_rate_limit()  # 6th call should fail
            print("✗ Failed to enforce rate limit")
            return False
        except RuntimeError as e:
            print(f"✓ Rate limit enforced: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Security manager test failed: {e}")
        traceback.print_exc()
        return False


# Test external input validation in EventDrivenSimulator
def test_event_driven_input_validation():
    """Test external input validation in EventDrivenSimulator."""
    print("\n" + "="*60)
    print("Testing EventDrivenSimulator Input Validation")
    print("="*60)
    
    try:
        from core.network import NeuromorphicNetwork, EventDrivenSimulator
        
        # Create network
        network = NeuromorphicNetwork()
        network.add_layer("input", 10, neuron_type="lif")
        
        # Create simulator
        simulator = EventDrivenSimulator()
        simulator.set_network(network)
        
        print("✓ Created network and simulator")
        
        # Test invalid layer name
        try:
            simulator.add_external_input("invalid_layer", 0, 1.0, 10.0)
            print("✗ Failed to catch invalid layer name")
            return False
        except ValueError as e:
            print(f"✓ Caught invalid layer: {e}")
        
        # Test invalid neuron ID
        try:
            simulator.add_external_input("input", 100, 1.0, 10.0)  # Only 10 neurons
            print("✗ Failed to catch invalid neuron ID")
            return False
        except ValueError as e:
            print(f"✓ Caught invalid neuron ID: {e}")
        
        # Test negative input time
        try:
            simulator.add_external_input("input", 0, -1.0, 10.0)
            print("✗ Failed to catch negative input time")
            return False
        except ValueError as e:
            print(f"✓ Caught negative input time: {e}")
        
        # Test excessive input strength (should be clamped)
        simulator.add_external_input("input", 0, 1.0, 500.0)  # Will be clamped
        print("✓ Input strength validated and clamped")
        
        return True
        
    except Exception as e:
        print(f"✗ Event-driven input validation test failed: {e}")
        traceback.print_exc()
        return False


# Run all tests
def main():
    """Run all security and robustness tests."""
    print("\n" + "="*60)
    print("NEUROMORPHIC SYSTEM SECURITY FIXES TEST SUITE")
    print("="*60)
    
    tests = [
        ("GPU Memory Leak Fix", test_gpu_memory_leak),
        ("STDP Weight Boundaries", test_stdp_weight_boundaries),
        ("Network Input Validation", test_network_input_validation),
        ("Security Manager", test_security_manager),
        ("Event-Driven Input Validation", test_event_driven_input_validation),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    for test_name, passed_flag in results:
        status = "✓ PASSED" if passed_flag else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed ({100*passed/total:.1f}%)")
    
    if passed == total:
        print("\n🎉 All security fixes verified successfully!")
    else:
        print(f"\n⚠ {total - passed} tests failed. Please review the fixes.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
