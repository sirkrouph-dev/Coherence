#!/usr/bin/env python3
"""
DIRECT CORE TEST - Bypass API, test core components directly
"""

import numpy as np
from core.network import NeuromorphicNetwork
from core.neurons import NeuronFactory

def test_core_directly():
    print("🔬 DIRECT CORE TEST")
    print("=" * 30)
    print("Testing core components without API...")
    
    # Test 1: Single neuron direct stimulation
    print(f"\n🧪 TEST 1: Single neuron stimulation")
    
    try:
        neuron = NeuronFactory.create_neuron('lif', neuron_id=0)
        print(f"✅ Created neuron: {type(neuron)}")
        
        # Test direct stimulation
        neuron.reset()
        dt = 0.1
        current = 25.0  # Strong current
        
        for step in range(200):
            spiked = neuron.step(dt, current)
            if spiked:
                print(f"✅ Neuron spiked at step {step} ({step * dt:.1f}ms)")
                break
        else:
            print(f"❌ Neuron never spiked with {current}nA")
            
    except Exception as e:
        print(f"❌ Neuron test error: {e}")
    
    # Test 2: Network layer direct test
    print(f"\n🧪 TEST 2: Network layer test")
    
    try:
        network = NeuromorphicNetwork()
        
        # Add layers directly
        network.add_layer("test_input", 2, "lif")
        network.add_layer("test_output", 1, "lif")
        
        # Connect layers
        network.connect_layers("test_input", "test_output", "stdp", 
                              connection_probability=1.0)
        
        print(f"✅ Created network: {list(network.layers.keys())}")
        
        # Try to inject external current directly
        # This should bypass all API input formatting issues
        
        # Manual neuron stimulation
        input_layer = network.layers["test_input"]
        print(f"Input layer size: {input_layer.size}")
        
        # Test if we can access and stimulate neurons
        layer_attrs = [attr for attr in dir(input_layer) if not attr.startswith('_')]
        print(f"Layer attributes: {layer_attrs[:5]}...")
        
        # Look for any pool-like attribute
        pool_attr = None
        for attr in ['neuron_pool', 'neurons', 'population', 'pool']:
            if hasattr(input_layer, attr):
                pool_attr = attr
                break
        
        if pool_attr:
            neuron_pool = getattr(input_layer, pool_attr)
            print(f"Found {pool_attr}: {type(neuron_pool)}")
            
            # Test direct pool stimulation
            if hasattr(neuron_pool, 'step'):
                print("Testing direct neuron pool stimulation...")
                
                # Create external current array
                external_currents = np.array([30.0, 25.0])  # Strong currents for 2 neurons
                
                # Step multiple times with strong current
                for step in range(100):
                    spike_indices, metrics = neuron_pool.step(0.1, external_currents)
                    
                    if len(spike_indices) > 0:
                        print(f"✅ Input spikes at step {step}: {spike_indices}")
                        break
                else:
                    print(f"❌ No input spikes with direct stimulation")
        else:
            print(f"❌ No neuron pool attribute found")
        
    except Exception as e:
        print(f"❌ Network test error: {e}")
    
    # Test 3: Manual network simulation
    print(f"\n🧪 TEST 3: Manual network step")
    
    try:
        network = NeuromorphicNetwork()
        network.add_layer("manual_test", 1, "lif")
        
        print("Testing manual network stepping...")
        
        # Get the layer and neuron
        test_layer = network.layers["manual_test"]
        
        # Try to step the network manually
        for step in range(50):
            # This should trigger neuron updates
            network.step(0.1)
            
            if step == 0:
                print(f"✅ Network step completed successfully")
                break
    
    except Exception as e:
        print(f"❌ Manual step error: {e}")

def test_simplest_possible():
    """The absolute simplest test possible"""
    print(f"\n🎯 SIMPLEST POSSIBLE TEST")
    print("-" * 25)
    
    try:
        # Create one neuron and make it spike
        from core.neurons import LeakyIntegrateAndFire
        
        neuron = LeakyIntegrateAndFire(neuron_id=0)
        print(f"Created LIF neuron")
        
        # Reset and stimulate strongly
        neuron.reset()
        
        # Very strong current
        for i in range(100):
            spiked = neuron.step(dt=0.1, I_syn=50.0)  # Correct parameter name
            if spiked:
                print(f"🎉 SUCCESS! Neuron spiked at step {i}")
                return True
        
        print(f"❌ Even simplest neuron won't spike")
        return False
        
    except Exception as e:
        print(f"❌ Simplest test failed: {e}")
        return False

def main():
    test_core_directly()
    success = test_simplest_possible()
    
    print(f"\n{'='*40}")
    print(f"🔍 CORE DIAGNOSIS")
    print(f"{'='*40}")
    
    if success:
        print(f"✅ Core neurons work - issue is in API/Network integration")
        print(f"🔧 Next: Fix external input handling in API")
    else:
        print(f"❌ Core neuron issue - check neuron parameters")
        print(f"🔧 Next: Debug neuron threshold/dynamics")

if __name__ == "__main__":
    main()
