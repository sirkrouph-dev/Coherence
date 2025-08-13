#!/usr/bin/env python3
"""
NETWORK SPIKE PROPAGATION TEST
Test if spikes are properly detected and passed through the network
"""

import numpy as np
from core.network import NeuromorphicNetwork

class SpikePropagationTest:
    def __init__(self):
        print("🔍 NETWORK SPIKE PROPAGATION TEST")
        print("=" * 35)
        
        self.network = NeuromorphicNetwork()
        self.setup_test_network()
        
    def setup_test_network(self):
        """Create test network"""
        self.network.add_layer("input", 2, "lif")
        self.network.add_layer("output", 1, "lif")
        
        self.network.connect_layers("input", "output", "stdp",
                                  connection_probability=1.0,
                                  weight=1.0,
                                  A_plus=0.5,  # Very strong for clear changes
                                  A_minus=0.2)
        
        print("✅ Test network: 2 → 1 neurons")
        
    def test_spike_detection_and_weight_updates(self):
        """Test if spikes are detected and weights update"""
        print("\n🧪 SPIKE DETECTION & WEIGHT UPDATE TEST")
        print("-" * 40)
        
        # Get initial weights
        initial_weights = []
        for (pre_layer, post_layer), connection in self.network.connections.items():
            for (pre_idx, post_idx), synapse in connection.synapse_population.synapses.items():
                initial_weights.append(synapse.weight)
        
        print(f"Initial weights: {[f'{w:.3f}' for w in initial_weights]}")
        
        # Get populations
        input_pop = self.network.layers["input"].neuron_population
        output_pop = self.network.layers["output"].neuron_population
        
        # Force spike generation and monitor
        dt = 0.1
        
        # Step 1: Generate input spike
        print("\nStep 1: Generating input spike...")
        input_states = input_pop.step(dt, [100.0, 0.0])  # Strong current to first neuron
        output_states = output_pop.step(dt, [0.0])
        
        print(f"Input spikes: {input_states}")
        print(f"Output spikes: {output_states}")
        
        # Network step - this should process spikes and update weights
        self.network.step(dt)
        
        # Step 2: Generate output spike with timing
        print("\nStep 2: Generating output spike...")
        input_states = input_pop.step(dt, [0.0, 0.0])
        output_states = output_pop.step(dt, [80.0])  # Strong current to output
        
        print(f"Input spikes: {input_states}")
        print(f"Output spikes: {output_states}")
        
        # Network step
        self.network.step(dt)
        
        # Check weights after spike sequence
        final_weights = []
        for (pre_layer, post_layer), connection in self.network.connections.items():
            for (pre_idx, post_idx), synapse in connection.synapse_population.synapses.items():
                final_weights.append(synapse.weight)
        
        print(f"Final weights: {[f'{w:.3f}' for w in final_weights]}")
        
        # Calculate changes
        weight_changes = [final - initial for initial, final in zip(initial_weights, final_weights)]
        print(f"Weight changes: {[f'{w:+.3f}' for w in weight_changes]}")
        
        return any(abs(change) > 0.001 for change in weight_changes)
    
    def test_multiple_spike_patterns(self):
        """Test multiple spike patterns to force learning"""
        print("\n🔄 MULTIPLE SPIKE PATTERN TEST")
        print("-" * 30)
        
        # Get initial weights
        initial_weights = []
        for (pre_layer, post_layer), connection in self.network.connections.items():
            for (pre_idx, post_idx), synapse in connection.synapse_population.synapses.items():
                initial_weights.append(synapse.weight)
        
        print(f"Starting weights: {[f'{w:.3f}' for w in initial_weights]}")
        
        input_pop = self.network.layers["input"].neuron_population
        output_pop = self.network.layers["output"].neuron_population
        
        # Pattern 1: Input[0] followed by Output - should strengthen synapse 0→0
        for i in range(5):  # Repeat pattern multiple times
            print(f"\nPattern 1, iteration {i+1}:")
            
            # Input spike
            input_states = input_pop.step(0.1, [100.0, 0.0])
            output_states = output_pop.step(0.1, [0.0])
            print(f"  Input step: {input_states}")
            self.network.step(0.1)
            
            # Small delay
            input_states = input_pop.step(0.1, [0.0, 0.0])
            output_states = output_pop.step(0.1, [0.0])
            self.network.step(0.1)
            
            # Output spike (slightly delayed)
            input_states = input_pop.step(0.1, [0.0, 0.0])
            output_states = output_pop.step(0.1, [100.0])
            print(f"  Output step: {output_states}")
            self.network.step(0.1)
            
            # Check weights after each iteration
            current_weights = []
            for (pre_layer, post_layer), connection in self.network.connections.items():
                for (pre_idx, post_idx), synapse in connection.synapse_population.synapses.items():
                    current_weights.append(synapse.weight)
            
            print(f"  Weights now: {[f'{w:.3f}' for w in current_weights]}")
        
        final_weights = current_weights
        weight_changes = [final - initial for initial, final in zip(initial_weights, final_weights)]
        
        print(f"\nFinal weight changes: {[f'{w:+.3f}' for w in weight_changes]}")
        
        return any(abs(change) > 0.01 for change in weight_changes)
    
    def run_tests(self):
        """Run all propagation tests"""
        print("Starting spike propagation tests...")
        
        test1_passed = self.test_spike_detection_and_weight_updates()
        test2_passed = self.test_multiple_spike_patterns()
        
        print(f"\n📊 TEST RESULTS")
        print("=" * 15)
        print(f"Single spike test: {'✅ PASS' if test1_passed else '❌ FAIL'}")
        print(f"Multiple pattern test: {'✅ PASS' if test2_passed else '❌ FAIL'}")
        
        if test1_passed or test2_passed:
            print(f"\n🎉 LEARNING IS WORKING! Weight changes detected.")
        else:
            print(f"\n❌ LEARNING NOT WORKING - No weight changes detected.")
        
        return test1_passed or test2_passed

if __name__ == "__main__":
    test = SpikePropagationTest()
    test.run_tests()
