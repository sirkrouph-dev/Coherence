#!/usr/bin/env python3
"""
Network Activity Diagnostic
Checking if the neuromorphic network is functioning correctly
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.network import NeuromorphicNetwork


def test_basic_network_activity():
    """Test if basic network activity is working"""
    print("🔍 NETWORK ACTIVITY DIAGNOSTIC")
    print("=" * 40)
    
    # Create minimal network
    network = NeuromorphicNetwork()
    network.add_layer("input", 4, "lif")
    network.add_layer("output", 2, "lif")
    network.connect_layers("input", "output", "stdp", 
                          connection_probability=1.0, weight=3.0)
    
    print("✅ Created 4->2 network with full connectivity")
    
    # Test 1: Check if input layer responds to stimulation
    print("\\n🧪 Test 1: Input layer responsiveness")
    input_currents = [50.0, 50.0, 0.0, 0.0]  # Strong stimulus
    
    for layer in network.layers.values():
        layer.neuron_population.reset()
    
    input_spikes_total = 0
    for step in range(50):
        input_spikes = network.layers["input"].neuron_population.step(0.1, input_currents)
        input_spikes_total += sum(input_spikes)
        
    print(f"   Input spikes in 50 steps: {input_spikes_total}")
    
    # Test 2: Check if connections propagate activity
    print("\\n🧪 Test 2: Connection propagation")
    for layer in network.layers.values():
        layer.neuron_population.reset()
    
    input_spikes_total = 0
    output_spikes_total = 0
    
    for step in range(100):  # Longer simulation
        input_spikes = network.layers["input"].neuron_population.step(0.1, input_currents)
        input_spikes_total += sum(input_spikes)
        
        # Let network process connections
        network.step(0.1)
        
        output_spikes = network.layers["output"].neuron_population.get_spike_states()
        output_spikes_total += sum(output_spikes)
        
    print(f"   Input spikes: {input_spikes_total}")
    print(f"   Output spikes: {output_spikes_total}")
    
    # Test 3: Check connection weights
    print("\\n🧪 Test 3: Connection weights")
    connection = network.connections[("input", "output")]
    if hasattr(connection, 'synapse_population') and connection.synapse_population:
        synapse_count = len(connection.synapse_population.synapses)
        print(f"   Total synapses: {synapse_count}")
        
        if synapse_count > 0:
            # Sample a few weights
            weights = []
            for i, ((pre, post), synapse) in enumerate(connection.synapse_population.synapses.items()):
                weights.append(synapse.weight)
                if i >= 5:  # Just sample first few
                    break
            print(f"   Sample weights: {weights[:5]}")
            print(f"   Average weight: {np.mean(list(weights)):.2f}")
    
    # Test 4: Manual current injection
    print("\\n🧪 Test 4: Manual output stimulation")
    for layer in network.layers.values():
        layer.neuron_population.reset()
        
    output_currents = [100.0, 100.0]  # Very strong direct stimulus
    output_spikes_total = 0
    
    for step in range(50):
        output_spikes = network.layers["output"].neuron_population.step(0.1, output_currents)
        output_spikes_total += sum(output_spikes)
        
    print(f"   Output spikes with direct stimulation: {output_spikes_total}")
    
    # Diagnosis
    print("\\n📋 DIAGNOSIS:")
    if input_spikes_total == 0:
        print("❌ Input layer not responding - neuron parameters may be wrong")
    elif output_spikes_total == 0 and output_spikes_total == 0:
        print("❌ No propagation - connection/synapse issue")
    elif output_spikes_total == 0:
        print("❌ Output layer not responding - check neuron parameters")
    else:
        print("✅ Basic network activity working")
        
    return {
        'input_spikes': input_spikes_total,
        'output_spikes': output_spikes_total,
        'direct_output_spikes': output_spikes_total
    }


def test_neuron_parameters():
    """Test individual neuron functionality"""
    print("\\n🧪 NEURON PARAMETER TEST")
    print("-" * 30)
    
    network = NeuromorphicNetwork()
    network.add_layer("test", 1, "lif")
    
    neuron = network.layers["test"].neuron_population
    
    # Test different current levels
    current_levels = [10.0, 30.0, 50.0, 100.0]
    
    for current in current_levels:
        neuron.reset()
        spikes = 0
        
        for step in range(100):
            spike_result = neuron.step(0.1, [current])
            spikes += sum(spike_result)
            
        print(f"   Current {current:5.1f}mA -> {spikes} spikes in 100 steps")


if __name__ == "__main__":
    test_basic_network_activity()
    test_neuron_parameters()
