"""
Basic functionality test for the neuromorphic programming system.
"""

import numpy as np
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_neuron_models():
    """Test basic neuron functionality."""
    print("Testing neuron models...")
    
    from core.neurons import AdaptiveExponentialIntegrateAndFire, LeakyIntegrateAndFire
    
    # Test AdEx neuron
    adex_neuron = AdaptiveExponentialIntegrateAndFire(0)
    spikes = []
    for i in range(100):
        # Inject current
        current = 1.0 if i < 50 else 0.0
        spiked = adex_neuron.step(0.1, current)
        if spiked:
            spikes.append(i * 0.1)
    
    print(f"AdEx neuron generated {len(spikes)} spikes")
    
    # Test LIF neuron
    lif_neuron = LeakyIntegrateAndFire(1)
    spikes = []
    for i in range(100):
        current = 1.0 if i < 50 else 0.0
        spiked = lif_neuron.step(0.1, current)
        if spiked:
            spikes.append(i * 0.1)
    
    print(f"LIF neuron generated {len(spikes)} spikes")
    
    return True

def test_synapse_models():
    """Test basic synapse functionality."""
    print("Testing synapse models...")
    
    from core.synapses import STDP_Synapse
    
    # Test STDP synapse
    synapse = STDP_Synapse(0, 0, 1)
    
    # Simulate pre-post spike pair (LTP)
    synapse.pre_spike(10.0)
    synapse.post_spike(12.0)
    initial_weight = synapse.weight
    
    # Simulate post-pre spike pair (LTD)
    synapse.pre_spike(20.0)
    synapse.post_spike(18.0)
    
    print(f"STDP synapse weight changed from {initial_weight:.3f} to {synapse.weight:.3f}")
    
    return True

def test_network_creation():
    """Test network creation and basic simulation."""
    print("Testing network creation...")
    
    from core.network import NeuromorphicNetwork
    
    # Create simple network
    network = NeuromorphicNetwork()
    network.add_layer("input", 10, "lif")
    network.add_layer("output", 5, "lif")
    network.connect_layers("input", "output", "stdp")
    
    # Run simulation
    results = network.run_simulation(50.0, 0.1)
    
    print(f"Network simulation completed")
    # Access network info directly from the network object
    network_info = network.get_network_info()
    print(f"Network has {network_info['total_neurons']} neurons")
    print(f"Network has {network_info['total_synapses']} synapses")
    
    return True

def test_sensory_encoding():
    """Test sensory encoding."""
    print("Testing sensory encoding...")
    
    from core.encoding import RetinalEncoder, RateEncoder
    
    # Test visual encoding
    visual_encoder = RetinalEncoder()
    image = np.random.rand(32, 32)
    visual_spikes = visual_encoder.encode(image)
    print(f"Visual encoding generated {len(visual_spikes)} spikes")
    
    # Test rate encoding
    rate_encoder = RateEncoder(10, 20)
    input_vector = np.random.rand(10)
    rate_spikes = rate_encoder.encode(input_vector)
    print(f"Rate encoding generated {len(rate_spikes)} spikes")
    
    return True

def test_api():
    """Test high-level API."""
    print("Testing high-level API...")
    
    from api.neuromorphic_api import NeuromorphicAPI
    
    # Create network using API
    api = NeuromorphicAPI()
    api.create_network()
    api.add_sensory_layer("input", 20, "rate")
    api.add_processing_layer("hidden", 10, "adex")
    api.add_motor_layer("output", 5)
    
    api.connect_layers("input", "hidden", "feedforward")
    api.connect_layers("hidden", "output", "feedforward")
    
    # Run simulation
    input_spikes = [(i, i * 5.0) for i in range(10)]
    results = api.run_simulation(50.0, external_inputs={"input": input_spikes})
    
    print(f"API simulation completed successfully")
    # Access network info from the API's network object
    network_info = api.network.get_network_info()
    print(f"Network info: {network_info}")
    
    return True

def main():
    """Run all tests."""
    print("Neuromorphic System Basic Functionality Test")
    print("=" * 50)
    
    tests = [
        test_neuron_models,
        test_synapse_models,
        test_network_creation,
        test_sensory_encoding,
        test_api
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
                print("✓ Test passed\n")
            else:
                print("✗ Test failed\n")
        except Exception as e:
            print(f"✗ Test failed with error: {e}\n")
    
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed! System is working correctly.")
        return True
    else:
        print("Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 