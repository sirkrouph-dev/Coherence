#!/usr/bin/env python3
"""
SPIKE PROPAGATION DEBUG - Find out why no spikes reach output
"""

import numpy as np
from api.neuromorphic_api import NeuromorphicAPI

def debug_spike_propagation():
    print("🔍 SPIKE PROPAGATION DEBUG")
    print("=" * 40)
    
    # Create minimal test network
    api = NeuromorphicAPI()
    api.create_network()
    
    # Simple: 4 input → 2 output (direct connection)
    api.add_sensory_layer("input", 4, "lif")
    api.add_motor_layer("output", 2)
    api.connect_layers("input", "output", "stdp", connection_probability=1.0)
    
    print("✅ Created: 4 input → 2 output (fully connected)")
    
    # Test 1: Single strong spike
    print(f"\n🧪 TEST 1: Single strong spike")
    single_spike = [(0, 10.0)]  # Neuron 0, time 10ms
    
    results = api.run_simulation(duration=50.0, external_inputs={"input": single_spike})
    
    for layer, spikes in results.get('layer_spike_times', {}).items():
        spike_count = sum(len(neuron_spikes) for neuron_spikes in spikes if isinstance(neuron_spikes, list))
        print(f"  {layer}: {spike_count} spikes")
        
        # Show individual neuron spikes
        if spike_count > 0:
            for i, neuron_spikes in enumerate(spikes):
                if isinstance(neuron_spikes, list) and len(neuron_spikes) > 0:
                    print(f"    Neuron {i}: {neuron_spikes}")
    
    # Test 2: Multiple strong spikes
    print(f"\n🧪 TEST 2: Multiple strong spikes")
    multi_spikes = [
        (0, 10.0), (0, 15.0), (0, 20.0),  # Neuron 0: burst
        (1, 12.0), (1, 17.0), (1, 22.0),  # Neuron 1: burst
    ]
    
    results = api.run_simulation(duration=50.0, external_inputs={"input": multi_spikes})
    
    for layer, spikes in results.get('layer_spike_times', {}).items():
        spike_count = sum(len(neuron_spikes) for neuron_spikes in spikes if isinstance(neuron_spikes, list))
        print(f"  {layer}: {spike_count} spikes")
    
    # Test 3: Check if synapses exist
    print(f"\n🧪 TEST 3: Network structure check")
    try:
        network = api.network
        if hasattr(network, 'layers'):
            print(f"  Layers: {list(network.layers.keys())}")
            for layer_name, layer in network.layers.items():
                print(f"    {layer_name}: {layer.size} neurons")
        
        if hasattr(network, 'connections'):
            print(f"  Connections: {len(network.connections)}")
            for i, conn in enumerate(network.connections):
                print(f"    Connection {i}: {conn.pre_layer_name} → {conn.post_layer_name}")
                if hasattr(conn, 'synapse_population'):
                    syn_pop = conn.synapse_population
                    print(f"      Synapses: {len(syn_pop.synapses) if hasattr(syn_pop, 'synapses') else 'unknown'}")
    
    except Exception as e:
        print(f"  ❌ Structure check error: {e}")
    
    # Test 4: Very high frequency spikes
    print(f"\n🧪 TEST 4: High frequency spike train")
    high_freq_spikes = []
    for neuron in range(4):
        for t in range(5, 45, 5):  # Every 5ms
            high_freq_spikes.append((neuron, float(t)))
    
    print(f"  Generated {len(high_freq_spikes)} input spikes")
    
    results = api.run_simulation(duration=50.0, external_inputs={"input": high_freq_spikes})
    
    for layer, spikes in results.get('layer_spike_times', {}).items():
        spike_count = sum(len(neuron_spikes) for neuron_spikes in spikes if isinstance(neuron_spikes, list))
        print(f"  {layer}: {spike_count} spikes")
        
        if layer == 'input' and spike_count > 0:
            print(f"    ✅ INPUT SPIKES WORKING!")
        elif layer == 'output' and spike_count > 0:
            print(f"    🎉 OUTPUT SPIKES DETECTED!")
        elif layer == 'output' and spike_count == 0:
            print(f"    ❌ No propagation to output")

def test_existing_demo():
    """Test if existing sensorimotor demo works"""
    print(f"\n🧪 TESTING EXISTING DEMO CODE")
    print("-" * 35)
    
    try:
        api = NeuromorphicAPI()
        api.create_network()
        
        api.add_sensory_layer("input", 10, "lif")
        api.add_processing_layer("hidden", 5, "lif")
        api.add_motor_layer("output", 3)
        
        api.connect_layers("input", "hidden", "stdp", connection_probability=0.5)
        api.connect_layers("hidden", "output", "stdp", connection_probability=0.7)
        
        # Use exact format from working demo
        input_spikes = [(i, i * 2.0) for i in range(5)]  # Like sensorimotor demo
        print(f"Input spikes: {input_spikes}")
        
        results = api.run_simulation(100.0, external_inputs={"input": input_spikes})
        
        print(f"Simulation duration: {results.get('duration', 'unknown')}")
        
        for layer_name, spike_times in results.get("layer_spike_times", {}).items():
            spike_count = sum(len(spikes) for spikes in spike_times if isinstance(spikes, list))
            print(f"{layer_name} layer: {spike_count} spikes")
            
            if spike_count > 0:
                print(f"  🎉 {layer_name} has activity!")
    
    except Exception as e:
        print(f"❌ Demo test error: {e}")

def main():
    debug_spike_propagation()
    test_existing_demo()
    
    print(f"\n{'='*50}")
    print(f"🔧 DIAGNOSIS SUMMARY")
    print(f"{'='*50}")
    print(f"If NO spikes reach output:")
    print(f"  1. Synaptic weights too weak")
    print(f"  2. Network not connected properly")
    print(f"  3. Neuron thresholds too high")
    print(f"  4. Time constants incompatible")

if __name__ == "__main__":
    main()
