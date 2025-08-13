#!/usr/bin/env python3
"""
MINIMAL SPIKE TEST - Let's get ONE neuron to spike first!
"""

import numpy as np
from api.neuromorphic_api import NeuromorphicAPI

def test_minimal_spiking():
    print("🔥 MINIMAL SPIKE TEST")
    print("=" * 40)
    print("Getting ONE neuron to spike...")
    
    # Create minimal network
    api = NeuromorphicAPI()
    api.create_network()
    
    # Just input and output
    api.add_sensory_layer("input", 4, "lif")
    api.add_motor_layer("output", 2)
    api.connect_layers("input", "output", "stdp", connection_probability=1.0)
    
    print("✅ Created: 4 input → 2 output neurons")
    
    # Test different external input formats
    test_formats = [
        # Format 1: Direct spike times (what API expects?)
        {
            "input": [(0, 10.0), (0, 20.0), (0, 30.0)]  # Neuron 0, times 10,20,30ms
        },
        
        # Format 2: Layer-based
        {
            "input": {
                0: [10.0, 20.0, 30.0],  # Neuron 0 spike times
                1: [15.0, 25.0, 35.0],  # Neuron 1 spike times
            }
        },
        
        # Format 3: Individual neuron naming
        {
            "input_0": [10.0, 20.0, 30.0],
            "input_1": [15.0, 25.0, 35.0],
        }
    ]
    
    for i, input_format in enumerate(test_formats):
        print(f"\n🧪 Testing format {i+1}: {type(list(input_format.values())[0])}")
        
        try:
            results = api.run_simulation(duration=50.0, external_inputs=input_format)
            
            if 'layer_spike_times' in results:
                for layer, spikes in results['layer_spike_times'].items():
                    spike_count = 0
                    if isinstance(spikes, list):
                        for neuron_spikes in spikes:
                            if isinstance(neuron_spikes, list):
                                spike_count += len(neuron_spikes)
                    
                    print(f"  {layer}: {spike_count} total spikes")
                    
                    if spike_count > 0:
                        print(f"  🎉 SUCCESS! Format {i+1} works!")
                        return input_format
            else:
                print(f"  ❌ No spike data returned")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print(f"\n❌ No format worked! Need to check API deeper...")
    return None

def test_direct_neuron_stimulation():
    """Test if we can stimulate neurons directly through the network"""
    print(f"\n🎯 DIRECT NEURON TEST")
    print("-" * 30)
    
    try:
        api = NeuromorphicAPI()
        api.create_network()
        api.add_sensory_layer("test", 1, "lif")
        
        # Try to access the network directly
        network = api.network
        if network and 'test' in network.layers:
            test_layer = network.layers['test']
            print(f"✅ Got test layer: {test_layer}")
            print(f"  Layer size: {test_layer.size}")
            
            # Try to access layer attributes safely
            layer_attrs = [attr for attr in dir(test_layer) if not attr.startswith('_')]
            print(f"  Layer attributes: {layer_attrs[:5]}...")
            
            # Test creating a standalone neuron instead
            from core.neurons import NeuronFactory
            test_neuron = NeuronFactory.create_neuron('lif', neuron_id=0)
            print(f"  Created test neuron: {type(test_neuron)}")
                
            # Try direct stimulation
            print(f"  Testing direct neuron stimulation...")
            for step in range(100):
                spiked = test_neuron.step(0.1, 25.0)  # dt, I_syn
                if spiked:
                    print(f"  🎉 Neuron spiked at step {step}!")
                    break
            else:
                print(f"  ❌ Neuron didn't spike with direct stimulation")
        
    except Exception as e:
        print(f"❌ Direct test error: {e}")

def main():
    working_format = test_minimal_spiking()
    test_direct_neuron_stimulation()
    
    if working_format:
        print(f"\n✅ Found working input format!")
        print(f"Use this format: {working_format}")
    else:
        print(f"\n🔧 Need to investigate API input handling...")

if __name__ == "__main__":
    main()
